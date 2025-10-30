"use client"

import { useState } from "react"
import { AppBar } from "@/components/app-bar"
import { ConfigForm } from "@/components/config-form"
import { Results } from "@/components/results"
import { ThemeProvider } from "@/components/theme-provider"
import type { RunRequest, RunResult } from "@/types/api"

export default function Home() {
  const [results, setResults] = useState<RunResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleRun = async (config: RunRequest) => {
    setIsLoading(true)
    try {
      const response = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      })

      if (!response.ok) {
        throw new Error("Failed to run benchmark")
      }

      const data: RunResult = await response.json()
      setResults(data)
    } catch (error) {
      console.error("[v0] Error running benchmark:", error)
      // Show error in results
      setResults({
        runs: [
          {
            runIndex: 0,
            duckdb: { error: "Failed to run benchmark" },
            sqlite: { error: "Failed to run benchmark" },
            qaoa: { supported: false, error: "Failed to run benchmark" },
          },
        ],
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <ThemeProvider>
      <div className="min-h-screen bg-gradient-to-br from-background via-background to-blue-50/30 dark:to-blue-950/20">
        <AppBar />
        <main className="container mx-auto px-4 py-6 max-w-6xl">
          <div className="space-y-6">
            <ConfigForm onRun={handleRun} isLoading={isLoading} />
            {results && <Results results={results} isLoading={isLoading} />}
          </div>
        </main>
      </div>
    </ThemeProvider>
  )
}
