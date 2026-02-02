const params = new URLSearchParams(self.location.search)
const scriptName = params.get("script") || "./obamify.js"

try {
  const obamifyModule = await import(scriptName)
  const wasmName = scriptName.replace(".js", "_bg.wasm")

  await obamifyModule.default(wasmName)
} catch (e) {
  console.error("worker failed to initialize:", e)
  throw e
}
