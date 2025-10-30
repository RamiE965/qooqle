import { NextResponse } from "next/server"
import { spawn } from "child_process"
import { writeFileSync, unlinkSync } from "fs"
import { join } from "path"

export async function GET() {
  try {
    // Create a simple test script
    const testScript = `
import json
print(json.dumps({"test": "success", "message": "Python is working"}))
`

    // Write the script to a temporary file
    const scriptPath = join(process.cwd(), 'test_script.py')
    writeFileSync(scriptPath, testScript)

    // Run the Python script
    const result = await new Promise<string>((resolve, reject) => {
      const python = spawn('/Users/nikhilsethuram/Documents/qooqle/venv/bin/python', [scriptPath], {
        cwd: '/Users/nikhilsethuram/Documents/qooqle'
      })

      let output = ''
      let error = ''

      python.stdout.on('data', (data) => {
        output += data.toString()
      })

      python.stderr.on('data', (data) => {
        error += data.toString()
      })

      python.on('close', (code) => {
        if (code === 0) {
          resolve(output)
        } else {
          reject(new Error(`Python script failed: ${error}`))
        }
      })
    })

    // Clean up the temporary file
    unlinkSync(scriptPath)

    return NextResponse.json({ result: result.trim() })

  } catch (error) {
    console.error("Test API error:", error)
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}


