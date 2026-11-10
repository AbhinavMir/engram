// Desktop shell for Engram. It starts server.py with the Python that the user chose and shows the page.
const { app, BrowserWindow, dialog, ipcMain, shell } = require('electron')
const { spawn } = require('node:child_process')
const fs = require('node:fs'), net = require('node:net'), os = require('node:os'), path = require('node:path')

const HOME = process.env.ENGRAM_HOME || path.join(os.homedir(), '.engram')
const CONFIG = path.join(HOME, 'app.json')
const packed = path.join(process.resourcesPath, 'engram')
const ROOT = fs.existsSync(path.join(packed, 'server.py')) ? packed : path.join(__dirname, '..')
const DEFAULT_PORT = 7860
let server = null, quitting = false, currentPort = DEFAULT_PORT

const readConfig = () => fs.existsSync(CONFIG) ? JSON.parse(fs.readFileSync(CONFIG, 'utf8')) : {}
const saveConfig = config => { fs.mkdirSync(HOME, { recursive: true, mode: 0o700 }); fs.writeFileSync(CONFIG, JSON.stringify(config, null, 2)) }
const validPort = n => Number.isInteger(n) && n >= 1024 && n <= 65535
const portFree = port => new Promise(ok => { const s = net.createServer(); s.on('error', () => ok(false)); s.listen(port, '127.0.0.1', () => s.close(() => ok(true))) })

async function choosePort(config) {
  const port = validPort(config.port) ? config.port : DEFAULT_PORT
  if (await portFree(port)) return port
  let next = port + 1
  while (next <= 65535 && !(await portFree(next))) next++
  if (next > 65535) { dialog.showErrorBox('Engram', `Port ${port} is in use and no higher port is free.`); return null }
  const { response } = await dialog.showMessageBox({ type: 'warning', message: `Port ${port} is in use by another program.`, buttons: [`Use port ${next}`, 'Quit'], defaultId: 0, cancelId: 1 })
  if (response !== 0) return null
  config.port = next; saveConfig(config)
  return next
}

async function python(config) {
  if (config.python && fs.existsSync(config.python)) return config.python
  const pick = await dialog.showOpenDialog({ title: 'Choose a Python that has torch, transformers, and safetensors', properties: ['openFile', 'showHiddenFiles'] })
  if (pick.canceled) return null
  config.python = pick.filePaths[0]
  saveConfig(config)
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
  const port = await choosePort(config)
  if (!port) return app.quit()
  currentPort = port
  const url = `http://127.0.0.1:${port}`
  const win = new BrowserWindow({ width: 1600, height: 950, title: 'Engram', webPreferences: { contextIsolation: true, nodeIntegration: false, preload: path.join(__dirname, 'preload.js') } })
  win.webContents.setWindowOpenHandler(({ url: u }) => { if (/^https:/.test(u)) shell.openExternal(u); return { action: 'deny' } })
  win.loadURL(waiting)
  const exportsDir = path.join(HOME, 'exports'); fs.mkdirSync(exportsDir, { recursive: true })
  const args = [path.join(ROOT, 'server.py'), '--port', String(port), ...(config.model ? ['--model', config.model] : [])]
  server = spawn(py, args, { cwd: ROOT, stdio: 'inherit', env: { ...process.env, ENGRAM_HOME: HOME, ENGRAM_EXPORTS: exportsDir, PYTHONDONTWRITEBYTECODE: '1' } })
  server.on('error', err => { if (!quitting) { dialog.showErrorBox('Engram', err.message); app.quit() } })
  try { await ready(url); await win.loadURL(url) }
  catch (err) { if (!quitting) { dialog.showErrorBox('Engram', err.message); app.quit() } }
}

if (!app.requestSingleInstanceLock()) app.quit()
app.on('second-instance', () => { const [win] = BrowserWindow.getAllWindows(); if (win) { if (win.isMinimized()) win.restore(); win.focus() } })
ipcMain.handle('port:get', () => currentPort)
ipcMain.handle('port:set', async (_event, port) => {
  if (!validPort(port)) return { error: 'Enter a port from 1024 to 65535.' }
  if (port === currentPort) return {}
  if (!(await portFree(port))) return { error: `Port ${port} is in use by another program.` }
  const config = readConfig(); config.port = port; saveConfig(config)
  app.relaunch(); app.quit()
  return {}
})
app.whenReady().then(start)
app.on('window-all-closed', () => app.quit())
app.on('before-quit', () => { quitting = true; if (server && server.exitCode === null) server.kill('SIGTERM') })
