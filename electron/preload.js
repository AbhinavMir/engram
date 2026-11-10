const { contextBridge, ipcRenderer } = require('electron')
contextBridge.exposeInMainWorld('engramDesktop', {
  getPort: () => ipcRenderer.invoke('port:get'),
  setPort: port => ipcRenderer.invoke('port:set', port),
})
