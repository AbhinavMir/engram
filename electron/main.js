// Desktop shell for Engram. It starts server.py with the Python that the user chose and shows the page.
const { app, BrowserWindow, dialog, shell } = require('electron')
const { spawn } = require('node:child_process')
const fs = require('node:fs'), net = require('node:net'), os = require('node:os'), path = require('node:path')

const HOME = process.env.ENGRAM_HOME || path.join(os.homedir(), '.engram')
const CONFIG = path.join(HOME, 'app.json')
const packed = path.join(process.resourcesPath, 'engram')
const ROOT = fs.existsSync(path.join(packed, 'server.py')) ? packed : path.join(__dirname, '..')
let server = null, quitting = false

const readConfig = () => fs.existsSync(CONFIG) ? JSON.parse(fs.readFileSync(CONFIG, 'utf8')) : {}
const freePort = () => new Promise((ok, no) => { const s = net.createServer(); s.on('error', no); s.listen(0, '127.0.0.1', () => { const { port } = s.address(); s.close(() => ok(port)) }) })

async function python(config) {
  if (config.python && fs.existsSync(config.python)) return config.python
  const pick = await dialog.showOpenDialog({ title: 'Choose a Python that has torch, transformers, and safetensors', properties: ['openFile', 'showHiddenFiles'] })
  if (pick.canceled) return null
  config.python = pick.filePaths[0]
  fs.mkdirSync(HOME, { recursive: true, mode: 0o700 }); fs.writeFileSync(CONFIG, JSON.stringify(config, null, 2))
  return config.python
}

const waiting = 'data:text/html,' + encodeURIComponent(`<!doctype html><title>Engram</title><style>body{margin:0;height:100vh;display:grid;place-items:center;font:15px -apple-system,sans-serif;color:%2354635a}i{display:block;width:26px;height:26px;margin:0 auto 12px;border-radius:50%;border:3px solid rgba(31,111,74,.22);border-top-color:%231f6f4a;animation:s .7s linear infinite}@keyframes s{to{transform:rotate(360deg)}}</style><div><i></i>Loading the model</div>`).replace(/%2523/g, '%23')

async function ready(url) {
  for (let i = 0; i < 720; i++) {
    if (server.exitCode !== null) throw new Error('The Engram server stopped. Make sure that the chosen Python has torch, transformers, and safetensors.')
    try { if ((await fetch(url + '/api/state')).ok) return } catch {}
    await new Promise(r => setTimeout(r, 250))
  }
  throw new Error('The Engram server did not start.')
}

async function start() {
  const config = readConfig(), py = await python(config)
  if (!py) return app.quit()
  const port = await freePort(), url = `http://127.0.0.1:${port}`
  const win = new BrowserWindow({ width: 1600, height: 950, title: 'Engram', webPreferences: { contextIsolation: true, nodeIntegration: false } })
  win.webContents.setWindowOpenHandler(({ url: u }) => { if (/^https:/.test(u)) shell.openExternal(u); return { action: 'deny' } })
  win.loadURL(waiting)
  const exportsDir = path.join(HOME, 'exports'); fs.mkdirSync(exportsDir, { recursive: true })
  const args = [path.join(ROOT, 'server.py'), '--port', String(port), ...(config.model ? ['--model', config.model] : [])]
  server = spawn(py, args, { cwd: ROOT, stdio: 'inherit', env: { ...process.env, ENGRAM_HOME: HOME, ENGRAM_EXPORTS: exportsDir, PYTHONDONTWRITEBYTECODE: '1' } })
  server.on('error', err => { if (!quitting) { dialog.showErrorBox('Engram', err.message); app.quit() } })
  try { await ready(url); await win.loadURL(url) }
  catch (err) { if (!quitting) { dialog.showErrorBox('Engram', err.message); app.quit() } }
}

app.whenReady().then(start)
app.on('window-all-closed', () => app.quit())
app.on('before-quit', () => { quitting = true; if (server && server.exitCode === null) server.kill('SIGTERM') })
