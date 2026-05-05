#!/usr/bin/env node

import { createHash } from 'node:crypto'
import { createInterface } from 'node:readline'

import { OpenMultiAgent } from '../src/orchestrator/orchestrator.js'
import type { SupportedProvider } from '../src/llm/adapter.js'
import type {
  AgentConfig,
  AgentRunResult,
  CoordinatorConfig,
  OrchestratorConfig,
  TeamRunResult,
  Task,
  TaskExecutionRecord,
  TeamConfig,
  TokenUsage,
} from '../src/types.js'

type JsonRpcId = string | number | null

interface JsonRpcRequest {
  jsonrpc: '2.0'
  id?: JsonRpcId
  method: string
  params?: unknown
}

interface PendingHostCall {
  method: string
  params: unknown
  startedAt: string
  resolve: (value: unknown) => void
  reject: (error: Error) => void
}

const DANGEROUS_TOOLS = [
  'bash',
  'file_write',
  'file_edit',
  'file_read',
  'grep',
  'glob',
  'delegate_to_agent',
] as const

const DEFAULT_MODEL = 'claude-opus-4-6'
const HOST_CALL_TIMEOUT_MS = 300_000
let nextHostCallId = 1
const pendingHostCalls = new Map<string | number, PendingHostCall>()

function writeMessage(message: unknown): void {
  process.stdout.write(`${JSON.stringify(message)}\n`)
}

function sendResult(id: JsonRpcId | undefined, result: unknown): void {
  if (id === undefined) return
  writeMessage({ jsonrpc: '2.0', id, result })
}

function sendError(id: JsonRpcId | undefined, code: number, message: string, data?: unknown): void {
  if (id === undefined) return
  writeMessage({
    jsonrpc: '2.0',
    id,
    error: {
      code,
      message,
      ...(data === undefined ? {} : { data }),
    },
  })
}

function notify(method: string, params: unknown): void {
  writeMessage({ jsonrpc: '2.0', method, params })
}

function debugLog(message: string, data?: unknown): void {
  const suffix = data === undefined ? '' : ` ${JSON.stringify(data)}`
  process.stderr.write(`[openma-sidecar] ${message}${suffix}\n`)
}

function requestHost(method: string, params: unknown): Promise<unknown> {
  const id = nextHostCallId++
  const startedAt = new Date().toISOString()
  debugLog('host-call request', { id, method })
  writeMessage({ jsonrpc: '2.0', id, method, params })
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      if (!pendingHostCalls.delete(id)) return
      reject(new Error(`Host-call timed out: ${method}`))
    }, HOST_CALL_TIMEOUT_MS)
    pendingHostCalls.set(id, {
      method,
      params,
      startedAt,
      resolve: (value) => {
        clearTimeout(timeout)
        resolve(value)
      },
      reject: (error) => {
        clearTimeout(timeout)
        reject(error)
      },
    })
  })
}

;(globalThis as typeof globalThis & { __opencrabHostCall?: typeof requestHost }).__opencrabHostCall = requestHost

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function asModel(value: unknown, provider?: SupportedProvider): string {
  if (typeof value === 'string' && value.trim()) return value.trim()
  if (provider === 'opencrab-codex') {
    throw new Error('OpenMA Codex model 未解析')
  }
  return DEFAULT_MODEL
}

function asProvider(value: unknown): SupportedProvider | undefined {
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim()
  return trimmed === 'opencrab-codex' ? trimmed : undefined
}

function assertOpenCrabCodexProvider(value: unknown): SupportedProvider {
  if (value !== 'opencrab-codex') {
    throw new Error('OpenCrab OpenMA execution requires provider: opencrab-codex')
  }
  return 'opencrab-codex'
}

function assertOpenCrabCodexModel(value: unknown): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error('OpenMA Codex model 未解析')
  }
  const model = value.trim()
  const lowered = model.toLowerCase()
  if (lowered === 'codex-default' || lowered === 'default' || lowered === 'placeholder') {
    throw new Error('OpenMA Codex model 未解析')
  }
  return model
}

function assertNoUnsafeTools(raw: Record<string, unknown>, label: string): void {
  if (Array.isArray(raw.tools) && raw.tools.length > 0) {
    throw new Error(`${label} must not configure OpenMA tools in text-only execution`)
  }
  if (raw.customTools !== undefined && raw.customTools !== null) {
    throw new Error(`${label} must not configure customTools in text-only execution`)
  }
  if (raw.toolPreset !== undefined && raw.toolPreset !== null) {
    throw new Error(`${label} must not configure toolPreset in text-only execution`)
  }
  const maybeMcp = raw.mcpTools ?? raw.mcp ?? raw.mcpServers
  if (maybeMcp !== undefined && maybeMcp !== null) {
    throw new Error(`${label} must not configure MCP tools in text-only execution`)
  }
}

function emptyToolAgent(agent: AgentConfig, fallbackModel: string): AgentConfig {
  return {
    ...agent,
    model: agent.model || fallbackModel,
    customTools: [],
    tools: [],
    disallowedTools: DANGEROUS_TOOLS,
    toolPreset: undefined,
  }
}

function forceOpenCrabCodexAgent(agent: AgentConfig, model: string): AgentConfig {
  return {
    ...emptyToolAgent(agent, model),
    provider: 'opencrab-codex',
    model: agent.model || model,
    apiKey: undefined,
    baseURL: undefined,
    customTools: [],
    tools: [],
    toolPreset: undefined,
    disallowedTools: DANGEROUS_TOOLS,
    maxTurns: 1,
  }
}

function defaultTeam(model: string, provider?: SupportedProvider): TeamConfig {
  return {
    name: 'opencrab-plan-team',
    sharedMemory: false,
    maxConcurrency: 1,
    agents: [
      {
        name: 'planner',
        model,
        ...(provider ? { provider } : {}),
        systemPrompt: 'You turn user goals into clear implementation planning tasks. Do not use tools.',
        tools: [],
        customTools: [],
        toolPreset: undefined,
        disallowedTools: DANGEROUS_TOOLS,
        maxTurns: 1,
      },
      {
        name: 'reviewer',
        model,
        ...(provider ? { provider } : {}),
        systemPrompt: 'You review plans for gaps, risks, and approval checkpoints. Do not use tools.',
        tools: [],
        customTools: [],
        toolPreset: undefined,
        disallowedTools: DANGEROUS_TOOLS,
        maxTurns: 1,
      },
    ],
  }
}

function sanitizeTeamConfig(raw: unknown, model: string): TeamConfig {
  if (!isRecord(raw)) return defaultTeam(model)

  const name = typeof raw.name === 'string' && raw.name.trim()
    ? raw.name.trim()
    : 'opencrab-plan-team'
  const rawAgents = Array.isArray(raw.agents) ? raw.agents : []
  const agents = rawAgents
    .filter(isRecord)
    .filter((agent) => typeof agent.name === 'string' && agent.name.trim())
    .map((agent) => emptyToolAgent(agent as unknown as AgentConfig, model))

  if (agents.length === 0) return defaultTeam(model)

  return {
    name,
    agents,
    sharedMemory: false,
    maxConcurrency: 1,
  }
}

function sanitizeExecutionTeamConfig(raw: unknown, model: string): TeamConfig {
  if (!isRecord(raw)) return defaultTeam(model, 'opencrab-codex')

  assertNoUnsafeTools(raw, 'teamConfig')
  const name = typeof raw.name === 'string' && raw.name.trim()
    ? raw.name.trim()
    : 'opencrab-execution-team'
  const rawAgents = Array.isArray(raw.agents) ? raw.agents : []
  const agents = rawAgents
    .filter(isRecord)
    .filter((agent) => typeof agent.name === 'string' && agent.name.trim())
    .map((agent) => {
      assertNoUnsafeTools(agent, `agent ${String(agent.name)}`)
      if (typeof agent.provider === 'string' && agent.provider !== 'opencrab-codex') {
        throw new Error(`agent ${String(agent.name)} must use provider opencrab-codex`)
      }
      return forceOpenCrabCodexAgent(agent as unknown as AgentConfig, model)
    })

  if (agents.length === 0) return defaultTeam(model, 'opencrab-codex')

  return {
    name,
    agents,
    sharedMemory: false,
    maxConcurrency: 1,
  }
}

function withExecutionExtraBody(teamConfig: TeamConfig, extraBody: Record<string, unknown>): TeamConfig {
  return {
    ...teamConfig,
    agents: teamConfig.agents.map((agent) => ({
      ...agent,
      extraBody: {
        ...(agent.extraBody ?? {}),
        ...extraBody,
        agent: agent.name,
        openmaAgent: agent.name,
      },
    })),
  }
}

function wrapGoalForPlanning(goal: string): string {
  return [
    'OpenCrab plan-only request. Decompose the user goal into a concrete multi-step task plan.',
    'Step 1: identify the essential work items.',
    'Step 2: assign each work item to the most suitable team member.',
    'Step 3: express dependencies explicitly.',
    'Step 4: stop at planning only; do not execute, inspect files, call tools, write files, patch code, run tests, or repair failures.',
    '',
    `User goal: ${goal}`,
  ].join('\n')
}

type HashableTask = {
  title: string
  description: string
  assignee: string | null
  dependsOn: string[]
}

function hashableTask(task: Task | Record<string, unknown>): HashableTask {
  const dependsOn = Array.isArray((task as { dependsOn?: unknown }).dependsOn)
    ? (task as { dependsOn: unknown[] }).dependsOn
    : Array.isArray((task as { depends_on?: unknown }).depends_on)
      ? (task as { depends_on: unknown[] }).depends_on
      : []
  return {
    title: typeof task.title === 'string' ? task.title : '',
    description: typeof task.description === 'string' ? task.description : '',
    assignee: typeof task.assignee === 'string' && task.assignee.trim() ? task.assignee : null,
    dependsOn: dependsOn.filter((item): item is string => typeof item === 'string'),
  }
}

function planHash(tasks: readonly (Task | Record<string, unknown>)[]): string {
  const canonical = tasks.map(hashableTask)
  const raw = JSON.stringify(canonical)
  return createHash('sha256').update(raw).digest('hex')
}

function approvedTaskSpecs(value: unknown): Array<{
  title: string
  description: string
  assignee?: string
  dependsOn?: string[]
}> {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error('run requires approvedTasks from the approved OpenMA plan')
  }
  const tasks = value.map((item, index) => {
    if (!isRecord(item)) {
      throw new Error(`approvedTasks[${index}] must be an object`)
    }
    if (typeof item.title !== 'string' || !item.title.trim()) {
      throw new Error(`approvedTasks[${index}] requires title`)
    }
    if (typeof item.description !== 'string' || !item.description.trim()) {
      throw new Error(`approvedTasks[${index}] requires description`)
    }
    const dependsOn = Array.isArray(item.dependsOn)
      ? item.dependsOn
      : Array.isArray(item.depends_on)
        ? item.depends_on
        : []
    return {
      title: item.title.trim(),
      description: item.description.trim(),
      assignee: typeof item.assignee === 'string' && item.assignee.trim()
        ? item.assignee.trim()
        : undefined,
      dependsOn: dependsOn.filter((entry): entry is string => typeof entry === 'string'),
    }
  })
  return tasks
}

function isPlanOnlyFallbackTask(task: Record<string, unknown>): boolean {
  const title = typeof task.title === 'string' ? task.title : ''
  const description = typeof task.description === 'string' ? task.description : ''
  return (
    description.includes('OpenCrab plan-only request.') ||
    title.startsWith('planner: OpenCrab plan-only request.') ||
    title.startsWith('reviewer: OpenCrab plan-only request.')
  )
}

function assertNoPlanOnlyFallbackTasks(
  tasks: readonly Record<string, unknown>[],
  phase: 'plan' | 'run',
): void {
  if (tasks.some(isPlanOnlyFallbackTask)) {
    throw new Error(
      `OpenMA ${phase} produced plan-only fallback tasks; regenerate the plan before execution`,
    )
  }
}

function serializeTask(task: Task): Record<string, unknown> {
  return {
    id: task.id,
    title: task.title,
    description: task.description,
    status: task.status,
    assignee: task.assignee,
    dependsOn: task.dependsOn ?? [],
    memoryScope: task.memoryScope,
    createdAt: task.createdAt instanceof Date ? task.createdAt.toISOString() : task.createdAt,
    updatedAt: task.updatedAt instanceof Date ? task.updatedAt.toISOString() : task.updatedAt,
    maxRetries: task.maxRetries,
    retryDelayMs: task.retryDelayMs,
    retryBackoff: task.retryBackoff,
  }
}

function serializeTaskRecord(task: TaskExecutionRecord): Record<string, unknown> {
  return {
    id: task.id,
    title: task.title,
    description: undefined,
    assignee: task.assignee,
    status: task.status,
    dependsOn: task.dependsOn,
    metrics: task.metrics,
  }
}

function serializeAgentResult(result: AgentRunResult): Record<string, unknown> {
  return {
    success: result.success,
    output: result.output,
    messages: result.messages,
    tokenUsage: result.tokenUsage,
    toolCalls: result.toolCalls,
    structured: result.structured,
    loopDetected: result.loopDetected,
    budgetExceeded: result.budgetExceeded,
  }
}

function serializeTeamRunResult(result: TeamRunResult): Record<string, unknown> {
  const agentResults: Record<string, unknown> = {}
  for (const [key, value] of result.agentResults) {
    agentResults[key] = serializeAgentResult(value)
  }
  const serialized: Record<string, unknown> = {
    success: result.success,
    tasks: result.tasks?.map(serializeTaskRecord) ?? [],
    totalTokenUsage: result.totalTokenUsage,
    agentResults,
  }
  if (typeof result.goal === 'string' && result.goal.trim()) {
    serialized.goal = result.goal
  }
  return serialized
}

function buildExecutedTasks(
  result: TeamRunResult,
  planTasks: readonly Record<string, unknown>[],
  taskOutputs: Map<string, string>,
): Record<string, unknown>[] {
  const planById = new Map(planTasks.map((task) => [String(task.id), task]))
  const planByTitle = new Map(planTasks.map((task) => [String(task.title), task]))
  return (result.tasks ?? []).map((task) => {
    const planned = planById.get(task.id) ?? planByTitle.get(task.title)
    return {
      ...serializeTaskRecord(task),
      title: typeof planned?.title === 'string' ? planned.title : task.title,
      description: typeof planned?.description === 'string' ? planned.description : '',
      output: taskOutputs.get(task.id),
      tokenUsage: task.metrics?.tokenUsage,
      toolCallsCount: task.metrics?.toolCalls.length ?? 0,
    }
  })
}

function finalResultFromTeamRun(result: TeamRunResult): string {
  const coordinator = result.agentResults.get('coordinator')
  if (coordinator?.output) return coordinator.output
  return Array.from(result.agentResults.values())
    .map((entry) => entry.output)
    .filter(Boolean)
    .join('\n\n---\n\n')
}

function assertNoToolCalls(result: TeamRunResult): void {
  for (const [agent, entry] of result.agentResults) {
    if (entry.toolCalls.length > 0) {
      throw new Error(`Text-only execution violated tool boundary: ${agent} made ${entry.toolCalls.length} tool call(s)`)
    }
  }
}

async function handlePlan(params: unknown): Promise<Record<string, unknown>> {
  if (!isRecord(params) || typeof params.goal !== 'string' || !params.goal.trim()) {
    throw new Error('plan requires params.goal as a non-empty string')
  }
  if (typeof params.runId !== 'string' || !params.runId.trim()) {
    throw new Error('plan requires params.runId as a non-empty string')
  }

  const runId = params.runId.trim()
  const goal = params.goal.trim()
  const provider = asProvider(params.provider)
  const model = provider === 'opencrab-codex'
    ? assertOpenCrabCodexModel(params.model)
    : asModel(params.model, provider)
  debugLog('plan start', { runId, provider: provider ?? 'default', model, goalChars: goal.length })
  const teamConfig = sanitizeTeamConfig(params.teamConfig, model)
  const extraBody = {
    runId,
    openmaRunId: runId,
  }
  let capturedTasks: Record<string, unknown>[] = []
  let planReadySeen = false
  let executionStarted = false

  const orchestratorConfig: OrchestratorConfig = {
    defaultModel: model,
    ...(provider ? { defaultProvider: provider } : {}),
    maxConcurrency: 1,
    onPlanReady: async (tasks) => {
      planReadySeen = true
      capturedTasks = tasks.map(serializeTask)
      debugLog('plan ready', { taskCount: capturedTasks.length, executionStarted: false })
      notify('openma/planReady', {
        runId,
        goal,
        tasks: capturedTasks,
        decisionRecorded: false,
        executionStarted: false,
      })
      return false
    },
    onProgress: (event) => {
      if (event.type === 'task_start' || (event.type === 'agent_start' && event.agent !== 'coordinator')) {
        executionStarted = true
      }
    },
  }

  const coordinator: CoordinatorConfig = {
    model,
    ...(provider ? { provider } : {}),
    tools: [],
    disallowedTools: DANGEROUS_TOOLS,
    maxTurns: 3,
    extraBody,
  }

  const orchestrator = new OpenMultiAgent(orchestratorConfig)
  let teamResult: TeamRunResult | undefined
  try {
    const team = orchestrator.createTeam(teamConfig.name, withExecutionExtraBody(teamConfig, extraBody))
    debugLog('plan runTeam begin', { team: teamConfig.name, agents: teamConfig.agents.map(agent => agent.name) })
    teamResult = await orchestrator.runTeam(team, wrapGoalForPlanning(goal), { coordinator })
    debugLog('plan runTeam returned', { planReadySeen, executionStarted })
  } finally {
    await orchestrator.shutdown()
  }

  const coordinatorResult = teamResult?.agentResults.get('coordinator:decompose')
  if (coordinatorResult && !coordinatorResult.success) {
    throw new Error(`OpenMA planner Codex completion failed: ${coordinatorResult.output}`)
  }

  if (!planReadySeen) {
    throw new Error('OpenMA did not emit onPlanReady; plan-only boundary was not reached')
  }
  assertNoPlanOnlyFallbackTasks(capturedTasks, 'plan')

  return {
    runId,
    goal,
    tasks: capturedTasks,
    decisionRecorded: false,
    executionStarted,
  }
}

async function handleRun(params: unknown): Promise<Record<string, unknown>> {
  if (!isRecord(params) || typeof params.runId !== 'string' || !params.runId.trim()) {
    throw new Error('run requires params.runId as a non-empty string')
  }
  if (typeof params.goal !== 'string' || !params.goal.trim()) {
    throw new Error('run requires params.goal as a non-empty string')
  }
  if (typeof params.approvedPlanHash !== 'string' || !params.approvedPlanHash.trim()) {
    throw new Error('run requires params.approvedPlanHash as a non-empty string')
  }

  const provider = assertOpenCrabCodexProvider(params.provider)
  const runId = params.runId.trim()
  const goal = params.goal.trim()
  const model = assertOpenCrabCodexModel(params.model)
  const workspaceId = typeof params.workspaceId === 'string' && params.workspaceId.trim()
    ? params.workspaceId.trim()
    : undefined
  const environmentRoot = typeof params.environmentRoot === 'string' && params.environmentRoot.trim()
    ? params.environmentRoot.trim()
    : undefined
  const approvedPlanHash = params.approvedPlanHash.trim()
  const approvedTasks = approvedTaskSpecs(params.approvedTasks)
  assertNoPlanOnlyFallbackTasks(approvedTasks, 'run')
  const executionPlanHash = planHash(approvedTasks)
  const planMatched = executionPlanHash === approvedPlanHash
  if (!planMatched) {
    const error = `OpenMA approved task hash mismatch: approved=${approvedPlanHash} tasks=${executionPlanHash}`
    notify('openma/planReady', {
      runId,
      goal,
      tasks: approvedTasks,
      approvedPlanHash,
      executionPlanHash,
      planMatched: false,
      decisionRecorded: true,
      executionStarted: false,
    })
    notify('openma/error', { runId, message: error })
    return {
      runId,
      goal,
      status: 'failed',
      provider,
      model,
      workspaceId,
      environmentRoot,
      approvedPlanHash,
      executionPlanHash,
      planHash: executionPlanHash,
      planMatched: false,
      executionStarted: false,
      tasks: approvedTasks,
      finalResult: '',
      totalTokenUsage: { input_tokens: 0, output_tokens: 0 } satisfies TokenUsage,
      agentResults: {},
      error,
    }
  }
  const teamConfig = sanitizeExecutionTeamConfig(params.teamConfig, model)
  let executionStarted = false
  const taskOutputs = new Map<string, string>()

  const extraBody = {
    runId,
    openmaRunId: runId,
    ...(workspaceId ? { workspaceId } : {}),
    ...(environmentRoot ? { environmentRoot } : {}),
  }

  const orchestratorConfig: OrchestratorConfig = {
    defaultModel: model,
    defaultProvider: provider,
    maxConcurrency: 1,
    onProgress: (event) => {
      if (event.type === 'task_start' || (event.type === 'agent_start' && event.agent !== 'coordinator')) {
        executionStarted = true
      }
      if (event.type === 'task_complete' && typeof event.task === 'string' && isRecord(event.data)) {
        const output = typeof event.data.output === 'string' ? event.data.output : undefined
        if (output) taskOutputs.set(event.task, output)
      }
      notify('openma/progress', { runId, event })
    },
    onTrace: (event) => {
      notify('openma/trace', { runId, event })
    },
    onAgentStream: (agentName, event) => {
      notify('openma/agentStream', { runId, agentName, event })
    },
  }

  const executionTeamConfig = withExecutionExtraBody(teamConfig, extraBody)
  const orchestrator = new OpenMultiAgent(orchestratorConfig)
  let teamResult: TeamRunResult
  notify('openma/runStarted', { runId, goal, provider, model, workspaceId, environmentRoot })
  notify('openma/planReady', {
    runId,
    goal,
    tasks: approvedTasks,
    approvedPlanHash,
    executionPlanHash,
    planMatched: true,
    decisionRecorded: true,
    executionStarted: false,
  })
  try {
    const team = orchestrator.createTeam(executionTeamConfig.name, executionTeamConfig)
    teamResult = await orchestrator.runTasks(team, approvedTasks)
  } finally {
    await orchestrator.shutdown()
  }

  assertNoToolCalls(teamResult)
  const serialized = serializeTeamRunResult(teamResult)
  const executedTasks = buildExecutedTasks(teamResult, approvedTasks, taskOutputs)
  const finalResult = finalResultFromTeamRun(teamResult)
  const status = teamResult.success ? 'completed' : 'failed'
  const response = {
    ...serialized,
    runId,
    goal,
    status,
    provider,
    model,
    workspaceId,
    environmentRoot,
    approvedPlanHash,
    executionPlanHash,
    planHash: executionPlanHash,
    planMatched,
    executionStarted,
    finalResult,
    tasks: executedTasks,
  }
  notify('openma/runCompleted', response)
  return response
}

async function handleRequest(request: JsonRpcRequest): Promise<void> {
  try {
    switch (request.method) {
      case 'initialize':
        sendResult(request.id, {
          name: 'openma-sidecar',
          version: '0.1.0',
          methods: ['initialize', 'plan', 'run', 'shutdown'],
        })
        return
      case 'plan':
        sendResult(request.id, await handlePlan(request.params))
        return
      case 'run':
        sendResult(request.id, await handleRun(request.params))
        return
      case 'shutdown':
        sendResult(request.id, { ok: true })
        setTimeout(() => process.exit(0), 0)
        return
      default:
        sendError(request.id, -32601, `Method not found: ${request.method}`)
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    notify('openma/error', { message })
    sendError(request.id, -32000, message)
  }
}

function handleHostResponse(message: Record<string, unknown>): boolean {
  const id = message.id
  if (typeof id !== 'string' && typeof id !== 'number') return false
  const pending = pendingHostCalls.get(id)
  if (!pending) return false
  pendingHostCalls.delete(id)

  if ('error' in message) {
    const error = isRecord(message.error) ? message.error : {}
    const text = typeof error.message === 'string' ? error.message : 'Host-call failed'
    debugLog('host-call error', { id, method: pending.method, message: text })
    pending.reject(new Error(text))
    return true
  }

  debugLog('host-call response', { id, method: pending.method })
  if (pending.method === 'opencrab/codexComplete' && isRecord(message.result)) {
    const params = isRecord(pending.params) ? pending.params : {}
    const usage = isRecord(message.result.usage) ? message.result.usage : {}
    notify('openma/codexEvidence', {
      callId: typeof params.callId === 'string' ? params.callId : String(id),
      runId: typeof params.runId === 'string' ? params.runId : undefined,
      taskId: typeof params.taskId === 'string' ? params.taskId : undefined,
      agent: typeof params.agent === 'string' ? params.agent : undefined,
      model: typeof message.result.model === 'string'
        ? message.result.model
        : typeof params.model === 'string'
          ? params.model
          : undefined,
      codexThreadId: typeof message.result.codexThreadId === 'string' ? message.result.codexThreadId : undefined,
      codexTurnId: typeof message.result.codexTurnId === 'string' ? message.result.codexTurnId : undefined,
      inputTokens: typeof usage.inputTokens === 'number' ? usage.inputTokens : undefined,
      outputTokens: typeof usage.outputTokens === 'number' ? usage.outputTokens : undefined,
      startedAt: pending.startedAt,
      completedAt: new Date().toISOString(),
      started_at: pending.startedAt,
      completed_at: new Date().toISOString(),
    })
  }

  pending.resolve(message.result)
  return true
}

const rl = createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
})

notify('openma/log', { message: 'openma sidecar ready' })

rl.on('line', (line) => {
  const trimmed = line.trim()
  if (!trimmed) return

  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch (error) {
    sendError(null, -32700, 'Parse error', error instanceof Error ? error.message : String(error))
    return
  }

  if (!isRecord(parsed) || parsed.jsonrpc !== '2.0') {
    const id = isRecord(parsed) ? (parsed.id as JsonRpcId | undefined) : null
    sendError(id, -32600, 'Invalid Request')
    return
  }

  if (handleHostResponse(parsed)) {
    return
  }

  if (typeof parsed.method !== 'string') {
    const id = parsed.id as JsonRpcId | undefined
    sendError(id, -32600, 'Invalid Request')
    return
  }

  void handleRequest(parsed as unknown as JsonRpcRequest)
})
