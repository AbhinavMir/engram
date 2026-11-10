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

const WIN = process.platform === 'win32'
const RUNTIME = path.join(HOME, 'runtime')
const runtimePython = WIN ? path.join(RUNTIME, 'Scripts', 'python.exe') : path.join(RUNTIME, 'bin', 'python')
const READY = path.join(RUNTIME, '.engram-ready')
const PACKAGES = ['torch', 'transformers', 'safetensors', 'huggingface_hub']

function uvPath() {
  const packed = path.join(process.resourcesPath, 'bin', WIN ? 'uv.exe' : 'uv')
  if (fs.existsSync(packed)) return packed
  const local = path.join(__dirname, 'bin', `${WIN ? 'win' : 'mac'}-${process.arch}`, WIN ? 'uv.exe' : 'uv')
  return fs.existsSync(local) ? local : 'uv'
}

const status = (win, text) => win.webContents.executeJavaScript(`document.getElementById('m').textContent = ${JSON.stringify(text)}`).catch(() => {})

function run(win, cmd, args, label) {
  return new Promise((ok, fail) => {
    const env = { ...process.env, UV_PYTHON_INSTALL_DIR: path.join(HOME, 'python'), UV_CACHE_DIR: path.join(HOME, 'uv-cache') }
    const p = spawn(cmd, args, { env })
    let tail = ''
    const take = d => { tail = (tail + d).slice(-2000); const line = String(d).trim().split(/\r?\n/).pop(); if (line) status(win, `${label}\n${line.slice(0, 120)}`) }
    p.stdout.on('data', take); p.stderr.on('data', take)
    p.on('error', fail)
    p.on('close', code => code === 0 ? ok() : fail(new Error(`${label} failed.\n${tail.trim().split(/\r?\n/).slice(-6).join('\n')}`)))
  })
}

async function installRuntime(win) {
  const uv = uvPath()
  await run(win, uv, ['venv', '--python', '3.12', '--allow-existing', RUNTIME], 'Installing Python 3.12')
  await run(win, uv, ['pip', 'install', '--python', runtimePython, ...PACKAGES], 'Installing PyTorch and Transformers')
  fs.writeFileSync(READY, PACKAGES.join('\n'))
}

async function python(config, win) {
  if (config.python && fs.existsSync(config.python)) return config.python
  if (fs.existsSync(READY) && fs.existsSync(runtimePython)) return runtimePython
  const { response } = await dialog.showMessageBox(win, {
    type: 'info', message: 'Engram needs Python with PyTorch.',
    detail: `Engram can install Python 3.12, PyTorch, and Transformers in ${RUNTIME}. The download is about 1 GB. You can also choose a Python that already has torch, transformers, and safetensors.`,
    buttons: ['Install', 'Choose a Python', 'Quit'], defaultId: 0, cancelId: 2,
  })
  if (response === 0) {
    try { await installRuntime(win); return runtimePython }
    catch (err) { dialog.showErrorBox('Engram', err.message); return null }
  }
  if (response !== 1) return null
  const pick = await dialog.showOpenDialog(win, { title: 'Choose a Python that has torch, transformers, and safetensors', properties: ['openFile', 'showHiddenFiles'] })
  if (pick.canceled) return null
  config.python = pick.filePaths[0]
  saveConfig(config)
  return config.python
}

const waiting = 'data:text/html,' + encodeURIComponent(`<!doctype html><title>Engram</title><style>body{margin:0;height:100vh;display:grid;place-items:center;font:15px -apple-system,sans-serif;color:%2354635a}i{display:block;width:26px;height:26px;margin:0 auto 12px;border-radius:50%;border:3px solid rgba(31,111,74,.22);border-top-color:%231f6f4a;animation:s .7s linear infinite}@keyframes s{to{transform:rotate(360deg)}}</style><div style="text-align:center;white-space:pre-line;max-width:640px"><i></i><span id=m>Starting Engram</span></div>`).replace(/%2523/g, '%23')

async function ready(url) {
  for (let i = 0; i < 720; i++) {
    if (server.exitCode !== null) throw new Error('The Engram server stopped. Make sure that the chosen Python has torch, transformers, and safetensors.')
    try { if ((await fetch(url + '/api/state')).ok) return } catch {}
    await new Promise(r => setTimeout(r, 250))
  }
  throw new Error('The Engram server did not start.')
}

async function start() {
  const config = readConfig()
  const win = new BrowserWindow({ width: 1600, height: 950, title: 'Engram', icon: path.join(ROOT, 'docs', 'logo.png'), webPreferences: { contextIsolation: true, nodeIntegration: false, preload: path.join(__dirname, 'preload.js') } })
  win.webContents.setWindowOpenHandler(({ url: u }) => { if (/^https:/.test(u)) shell.openExternal(u); return { action: 'deny' } })
  await win.loadURL(waiting)
  const py = await python(config, win)
  if (!py) return app.quit()
  const port = await choosePort(config)
  if (!port) return app.quit()
  currentPort = port
  const url = `http://127.0.0.1:${port}`
  status(win, 'Starting the Engram server')
  const exportsDir = path.join(HOME, 'exports'); fs.mkdirSync(exportsDir, { recursive: true })
  const args = [path.join(ROOT, 'server.py'), '--port', String(port), ...(config.model ? ['--model', config.model] : [])]
  // ENGRAM_NODE lets server.py run the AI process on the Node runtime inside Electron.
  const env = { ...process.env, ENGRAM_HOME: HOME, ENGRAM_EXPORTS: exportsDir, ENGRAM_NODE: process.execPath, ELECTRON_RUN_AS_NODE: '1', PYTHONDONTWRITEBYTECODE: '1' }
  server = spawn(py, args, { cwd: ROOT, stdio: 'inherit', env, windowsHide: true })
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
