/**
 * @fileoverview OpenCrab/Codex-backed LLM adapter for OpenMA sidecar runs.
 *
 * This adapter intentionally implements only text-only model completion. Tool
 * execution stays outside the LLM channel and must go through a separate,
 * approval-aware OpenCrab capability path in a later milestone.
 */

import type {
  ContentBlock,
  LLMAdapter,
  LLMChatOptions,
  LLMMessage,
  LLMResponse,
  LLMStreamOptions,
  StreamEvent,
  TextBlock,
  TokenUsage,
} from '../types.js'

type HostCall = (method: string, params: unknown) => Promise<unknown>

interface CodexCompleteMessage {
  role: 'user' | 'assistant'
  content: string
}

interface CodexCompleteResponse {
  id: string
  model?: string
  text: string
  stopReason?: 'end_turn' | 'max_tokens' | 'error' | 'unknown'
  usage?: {
    inputTokens?: number
    outputTokens?: number
  }
  codexThreadId?: string
  codexTurnId?: string
}

declare global {
  // Installed by the OpenMA sidecar. Kept deliberately narrow: the adapter can
  // request a Codex completion, not arbitrary host tools.
  // eslint-disable-next-line no-var
  var __opencrabHostCall: HostCall | undefined
}

function textFromBlocks(blocks: readonly ContentBlock[]): string {
  const parts: string[] = []
  for (const block of blocks) {
    if (block.type === 'text' || block.type === 'reasoning') {
      parts.push(block.text)
      continue
    }
    if (block.type === 'tool_use' || block.type === 'tool_result') {
      throw new Error('opencrab-codex does not support OpenMA tool blocks in LLM messages')
    }
    if (block.type === 'image') {
      throw new Error('opencrab-codex only supports text messages in the MVP')
    }
  }
  return parts.join('')
}

function toCodexMessages(messages: readonly LLMMessage[]): CodexCompleteMessage[] {
  return messages.map((message) => ({
    role: message.role,
    content: textFromBlocks(message.content),
  }))
}

function usageFromResponse(response: CodexCompleteResponse): TokenUsage {
  return {
    input_tokens: response.usage?.inputTokens ?? 0,
    output_tokens: response.usage?.outputTokens ?? 0,
  }
}

function randomCallId(): string {
  const random = Math.random().toString(36).slice(2)
  return `openma_codex_${Date.now()}_${random}`
}

function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function debugLog(message: string, data?: unknown): void {
  const suffix = data === undefined ? '' : ` ${JSON.stringify(data)}`
  process.stderr.write(`[openma-codex-adapter] ${message}${suffix}\n`)
}

/**
 * OpenMA adapter backed by OpenCrab's existing Codex CLI / app-server channel.
 */
export class OpenMaCodexAdapter implements LLMAdapter {
  readonly name = 'opencrab-codex'

  async chat(messages: LLMMessage[], options: LLMChatOptions): Promise<LLMResponse> {
    if (options.tools && options.tools.length > 0) {
      throw new Error('opencrab-codex MVP does not support LLM tool definitions')
    }

    const hostCall = globalThis.__opencrabHostCall
    if (!hostCall) {
      throw new Error('opencrab-codex host-call bridge is unavailable')
    }

    const extra = options.extraBody ?? {}
    const callId = optionalString(extra['callId']) ?? randomCallId()
    debugLog('chat request', {
      callId,
      model: options.model,
      messageCount: messages.length,
      hasSystemPrompt: Boolean(options.systemPrompt),
      runId: optionalString(extra['runId'] ?? extra['openmaRunId']),
      agent: optionalString(extra['agent'] ?? extra['openmaAgent']),
    })
    const response = await hostCall('opencrab/codexComplete', {
      workspaceId: optionalString(extra['workspaceId'] ?? extra['opencrabWorkspaceId']),
      runId: optionalString(extra['runId'] ?? extra['openmaRunId']),
      taskId: optionalString(extra['taskId'] ?? extra['openmaTaskId']),
      agent: optionalString(extra['agent'] ?? extra['openmaAgent']),
      callId,
      model: options.model,
      systemPrompt: options.systemPrompt,
      messages: toCodexMessages(messages),
      maxTokens: options.maxTokens,
      temperature: options.temperature,
    })

    if (!response || typeof response !== 'object' || Array.isArray(response)) {
      throw new Error('opencrab-codex host returned an invalid response')
    }
    const result = response as Partial<CodexCompleteResponse>
    if (typeof result.text !== 'string') {
      throw new Error('opencrab-codex host response missing text')
    }
    debugLog('chat response', {
      callId,
      textChars: result.text.length,
      codexThreadId: result.codexThreadId,
      codexTurnId: result.codexTurnId,
    })

    const textBlock: TextBlock = { type: 'text', text: result.text }
    return {
      id: typeof result.id === 'string' ? result.id : callId,
      content: [textBlock],
      model: result.model ?? options.model,
      stop_reason: result.stopReason ?? 'end_turn',
      usage: usageFromResponse(result as CodexCompleteResponse),
    }
  }

  async *stream(
    messages: LLMMessage[],
    options: LLMStreamOptions,
  ): AsyncIterable<StreamEvent> {
    try {
      const response = await this.chat(messages, options)
      const text = response.content
        .filter((block): block is TextBlock => block.type === 'text')
        .map(block => block.text)
        .join('')
      if (text.length > 0) {
        yield { type: 'text', data: text }
      }
      yield { type: 'done', data: response }
    } catch (error) {
      yield { type: 'error', data: error instanceof Error ? error : new Error(String(error)) }
    }
  }
}
