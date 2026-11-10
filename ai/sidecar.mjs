// AI side of Engram. server.py starts this process and is its only client.
// Provider list and key storage follow the same design as Soquel: local servers first, one key per provider, file mode 0600.
import { generateText } from 'ai'
import { createAnthropic } from '@ai-sdk/anthropic'
import { createOpenAI } from '@ai-sdk/openai'
import { createOpenAICompatible } from '@ai-sdk/openai-compatible'
import { z } from 'zod'
import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

const DIR = process.env.ENGRAM_HOME || join(homedir(), '.engram')
const SETTINGS = join(DIR, 'ai.json'), KEYS = join(DIR, 'credentials.json')
mkdirSync(DIR, { recursive: true, mode: 0o700 })
const read = p => existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')) : {}
const write = (p, v) => { writeFileSync(p, JSON.stringify(v, null, 2), { mode: 0o600 }); chmodSync(p, 0o600) }

const PRESETS = [
  { id: 'ollama', name: 'Ollama (this computer)', base: 'http://localhost:11434/v1', sdk: 'compatible', local: true, needsKey: false, model: '' },
  { id: 'lmstudio', name: 'LM Studio (this computer)', base: 'http://localhost:1234/v1', sdk: 'compatible', local: true, needsKey: false, model: '' },
  { id: 'llamacpp', name: 'llama.cpp server (this computer)', base: 'http://localhost:8080/v1', sdk: 'compatible', local: true, needsKey: false, model: '' },
  { id: 'anthropic', name: 'Anthropic', base: 'https://api.anthropic.com/v1', sdk: 'anthropic', local: false, needsKey: true, model: 'claude-opus-5-5', keyUrl: 'https://console.anthropic.com/settings/keys' },
  { id: 'openai', name: 'OpenAI', base: 'https://api.openai.com/v1', sdk: 'openai', local: false, needsKey: true, model: '', keyUrl: 'https://platform.openai.com/api-keys' },
  { id: 'openrouter', name: 'OpenRouter', base: 'https://openrouter.ai/api/v1', sdk: 'compatible', local: false, needsKey: true, model: '', keyUrl: 'https://openrouter.ai/keys' },
  { id: 'groq', name: 'Groq', base: 'https://api.groq.com/openai/v1', sdk: 'compatible', local: false, needsKey: true, model: '', keyUrl: 'https://console.groq.com/keys' },
  { id: 'together', name: 'Together', base: 'https://api.together.xyz/v1', sdk: 'compatible', local: false, needsKey: true, model: '', keyUrl: 'https://api.together.xyz/settings/api-keys' },
  { id: 'deepseek', name: 'DeepSeek', base: 'https://api.deepseek.com/v1', sdk: 'compatible', local: false, needsKey: true, model: 'deepseek-chat', keyUrl: 'https://platform.deepseek.com' },
  { id: 'custom', name: 'Other address', base: '', sdk: 'compatible', local: false, needsKey: false, model: '' },
]
const preset = id => PRESETS.find(p => p.id === id)
const state = () => { const s = read(SETTINGS); return { provider: s.provider || 'ollama', models: s.models || {}, endpoints: s.endpoints || {}, wire: s.wire || 'compatible' } }

function languageModel() {
  const s = state(), p = preset(s.provider), base = s.endpoints[p.id] || p.base, modelId = s.models[p.id] || p.model, key = read(KEYS)[p.id]
  if (!base) throw new Error('Enter the address of this provider.')
  if (!modelId) throw new Error('Choose a model for this provider.')
  if (p.needsKey && !key) throw new Error(`Enter an API key for ${p.name}.`)
  const sdk = p.id === 'custom' ? s.wire : p.sdk
  if (sdk === 'anthropic') return createAnthropic({ apiKey: key || 'none', baseURL: base })(modelId)
  if (sdk === 'openai') return createOpenAI({ apiKey: key, baseURL: base }).chat(modelId)
  return createOpenAICompatible({ name: p.id, baseURL: base, apiKey: key || undefined })(modelId)
}

const why = z.string().max(400)
const Action = z.discriminatedUnion('action', [
  z.object({ action: z.literal('rank_one'), layer: z.number().int().min(0), component: z.string(), alpha: z.number().min(-10).max(10).refine(v => v !== 0), target: z.string().min(1), foil: z.string().min(1), position: z.number().int().min(0), why }),
  z.object({ action: z.literal('scale'), layer: z.number().int().min(0), component: z.string(), rows: z.string(), cols: z.string(), factor: z.number().min(-2).max(2), why }),
  z.object({ action: z.literal('remove_edit'), id: z.number().int(), why }),
  z.object({ action: z.literal('finish'), success: z.boolean(), summary: z.string().max(600) }),
])

const SYSTEM = `You edit the weights of a small language model through the Engram tool, and you work one step at a time.
Reply with one JSON object and no other text. Use one of these forms:
{"action":"rank_one","layer":18,"component":"mlp.down_proj","alpha":0.5,"target":" Berlin","foil":" Paris","position":4,"why":"..."}
  This adds alpha*u*v^T to one matrix. u moves the output from foil to target. v is the input at the token position.
{"action":"scale","layer":10,"component":"self_attn.o_proj","rows":"","cols":"64:128","factor":0,"why":"..."}
  This multiplies the entries in a row and column range. An empty range means all. Factor 0 clears the entries.
{"action":"remove_edit","id":3,"why":"..."}
  This deletes one edit that you can see in state.edits.
{"action":"finish","success":true,"summary":"..."}
  This ends the work.
Rules:
- target and foil must be single tokens exactly as the tokens appear in the state, with a leading space if the token has one.
- To change an answer, use rank_one at the last prompt token, on a layer near three quarters of the model depth, with a small alpha such as 0.5.
- The answer changes in state.answer_edited after each edit. Compare it with state.answer_base.
- If an edit had no effect, remove it and try a larger alpha or a later layer. Do not stack several tries.
- If the output repeats one word or stops making sense, alpha is too large. Remove the edit and use a smaller alpha.
- Choose the smallest edit that reaches the goal, then finish.
- If the goal cannot be reached, finish with success false and say why.`

function parseAction(text) {
  const clean = text.replace(/<think>[\s\S]*?<\/think>/g, ''), a = clean.indexOf('{'), b = clean.lastIndexOf('}')
  if (a < 0 || b <= a) throw new Error('The reply has no JSON object.')
  return Action.parse(JSON.parse(clean.slice(a, b + 1)))
}

async function step({ goal, state: bench, history, write: canWrite }) {
  const model = languageModel()
  const messages = [{ role: 'user', content: JSON.stringify({ goal, may_apply_edits: canWrite, state: bench, previous_steps: history }) }]
  let last
  for (let i = 0; i < 2; i++) {
    const { text } = await generateText({ model, system: SYSTEM, messages, maxRetries: 1 })
    try { return { action: parseAction(text) } }
    catch (err) {
      last = err instanceof z.ZodError ? err.issues.map(x => `${x.path.join('.')}: ${x.message}`).join('; ') : err.message
      messages.push({ role: 'assistant', content: text }, { role: 'user', content: `That reply was not valid: ${last}. Send one corrected JSON object.` })
    }
  }
  throw new Error(`The model sent an invalid action: ${last}`)
}

async function local() {
  const out = {}
  await Promise.all(PRESETS.filter(p => p.local).map(async p => {
    try {
      const r = await fetch(`${state().endpoints[p.id] || p.base}/models`, { signal: AbortSignal.timeout(800) })
      if (!r.ok) throw new Error(String(r.status))
      out[p.id] = { running: true, models: ((await r.json()).data || []).map(m => m.id) }
    } catch { out[p.id] = { running: false, models: [] } }
  }))
  return out
}

function publicSettings() { const keys = read(KEYS); return { ...state(), presets: PRESETS, hasKey: Object.fromEntries(PRESETS.map(p => [p.id, !!keys[p.id]])), keyFile: KEYS } }

function save(b) {
  const s = state()
  if (b.provider) { if (!preset(b.provider)) throw new Error('Unknown provider.'); s.provider = b.provider }
  if (b.model !== undefined) s.models[s.provider] = b.model
  if (b.endpoint !== undefined) { if (b.endpoint) s.endpoints[s.provider] = b.endpoint.replace(/\/+$/, ''); else delete s.endpoints[s.provider] }
  if (b.wire) s.wire = b.wire
  write(SETTINGS, s)
  if (b.key !== undefined || b.clearKey) { const k = read(KEYS); if (b.clearKey) delete k[s.provider]; else k[s.provider] = b.key; write(KEYS, k) }
  return publicSettings()
}

Bun.serve({
  hostname: '127.0.0.1', port: Number(process.env.PORT), idleTimeout: 255,
  async fetch(req) {
    const path = new URL(req.url).pathname
    try {
      let out
      if (path === '/settings' && req.method === 'GET') out = publicSettings()
      else if (path === '/settings') out = save(await req.json())
      else if (path === '/local') out = await local()
      else if (path === '/step') out = await step(await req.json())
      else return Response.json({ error: 'Unknown route.' }, { status: 404 })
      return Response.json(out)
    } catch (err) { return Response.json({ error: err.message || String(err) }, { status: 400 }) }
  },
})
