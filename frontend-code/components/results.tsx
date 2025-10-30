"use client"

import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import { StatCard } from "./stat-card"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "./ui/accordion"
import { Button } from "./ui/button"
import { Copy, Check, Trophy, Clock, Zap } from "lucide-react"
import { useState } from "react"
import type { RunResult } from "@/types/api"

export function Results({ results, isLoading }: { results: RunResult; isLoading: boolean }) {
  const [copiedSql, setCopiedSql] = useState(false)
  const [copiedExplain, setCopiedExplain] = useState<Record<string, boolean>>({})

  // Get the first run for display
  const run = results.runs[0]

  const copyToClipboard = async (text: string, type: "sql" | string) => {
    await navigator.clipboard.writeText(text)
    if (type === "sql") {
      setCopiedSql(true)
      setTimeout(() => setCopiedSql(false), 2000)
    } else {
      setCopiedExplain({ ...copiedExplain, [type]: true })
      setTimeout(() => setCopiedExplain({ ...copiedExplain, [type]: false }), 2000)
    }
  }

  // Determine winner based on execution time (use execTimeMs for QAOA, timeMs for others)
  const getWinner = () => {
    const engines = [
      { name: 'DuckDB', data: run.duckdb },
      { name: 'SQLite', data: run.sqlite },
      { name: 'QAOA', data: run.qaoa }
    ].filter(engine => engine.data && !engine.data.error)

    if (engines.length === 0) return null

    return engines.reduce((winner, current) => {
      // For QAOA, use execTimeMs (pure quantum execution time)
      // For others, use timeMs (execution time only)
      const winnerTime = winner.name === 'QAOA' 
        ? (winner.data.execTimeMs || Infinity)
        : (winner.data.timeMs || Infinity)
      
      const currentTime = current.name === 'QAOA'
        ? (current.data.execTimeMs || Infinity)
        : (current.data.timeMs || Infinity)
      
      return currentTime < winnerTime ? current : winner
    })
  }

  const winner = getWinner()

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle className="text-lg font-medium flex items-center gap-2">
          <Trophy className="h-5 w-5" />
          Optimization Results
          {winner && (
            <span className="text-sm font-normal text-green-600 ml-auto">
              🏆 {winner.name} Wins!
            </span>
          )}
        </CardTitle>
        {isLoading && (
          <div className="h-1 w-full bg-muted rounded-full overflow-hidden">
            <div className="h-full bg-primary animate-[shimmer_1s_ease-in-out_infinite] w-1/3" />
          </div>
        )}
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Performance Summary */}
        {winner && (
          <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-950/20 dark:to-blue-950/20 p-4 rounded-lg border">
            <div className="flex items-center gap-2 mb-2">
              <Trophy className="h-4 w-4 text-yellow-600" />
              <span className="font-medium text-sm">Best Performance</span>
            </div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              <strong>{winner.name}</strong> achieved the fastest execution time
              {winner.name === 'QAOA' 
                ? ` (${winner.data.execTimeMs?.toFixed(2)}ms execution)`
                : ` (${winner.data.timeMs?.toFixed(2)}ms)`
              }
            </div>
          </div>
        )}

        {/* Stat Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <StatCard title="DuckDB" data={run.duckdb} />
          <StatCard title="SQLite" data={run.sqlite} />
          <StatCard title="QAOA" data={run.qaoa} />
        </div>

        {/* Detailed Results */}
        <Accordion type="multiple" className="w-full">
          {/* DuckDB Details */}
          {run.duckdb && !run.duckdb.error && (
            <AccordionItem value="duckdb-details">
              <AccordionTrigger className="text-sm font-medium flex items-center gap-2">
                <Clock className="h-4 w-4" />
                DuckDB Details
              </AccordionTrigger>
              <AccordionContent>
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Execution Time:</span>
                      <span className="ml-2">{run.duckdb.timeMs?.toFixed(2)}ms</span>
                    </div>
                    <div>
                      <span className="font-medium">Estimated Cost:</span>
                      <span className="ml-2">{run.duckdb.cost?.toLocaleString()}</span>
                    </div>
                    <div className="col-span-2">
                      <span className="font-medium">Join Order:</span>
                      <span className="ml-2 font-mono">{run.duckdb.joinOrder}</span>
                    </div>
                  </div>
                  {run.duckdb.explain && (
                    <div className="relative">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="absolute top-2 right-2 h-8 w-8"
                        onClick={() => copyToClipboard(run.duckdb!.explain!, "duckdb")}
                      >
                        {copiedExplain.duckdb ? <Check className="h-4 w-4 text-green-600" /> : <Copy className="h-4 w-4" />}
                      </Button>
                      <pre className="bg-muted p-4 rounded-lg text-xs font-mono overflow-x-auto whitespace-pre-wrap">
                        {run.duckdb.explain}
                      </pre>
                    </div>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>
          )}

          {/* SQLite Details */}
          {run.sqlite && !run.sqlite.error && (
            <AccordionItem value="sqlite-details">
              <AccordionTrigger className="text-sm font-medium flex items-center gap-2">
                <Clock className="h-4 w-4" />
                SQLite Details
              </AccordionTrigger>
              <AccordionContent>
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Execution Time:</span>
                      <span className="ml-2">{run.sqlite.timeMs?.toFixed(2)}ms</span>
                    </div>
                    <div>
                      <span className="font-medium">Estimated Cost:</span>
                      <span className="ml-2">{run.sqlite.cost?.toLocaleString()}</span>
                    </div>
                    <div className="col-span-2">
                      <span className="font-medium">Join Order:</span>
                      <span className="ml-2 font-mono">{run.sqlite.joinOrder}</span>
                    </div>
                  </div>
                  {run.sqlite.explain && (
                    <div className="relative">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="absolute top-2 right-2 h-8 w-8"
                        onClick={() => copyToClipboard(run.sqlite!.explain!, "sqlite")}
                      >
                        {copiedExplain.sqlite ? <Check className="h-4 w-4 text-green-600" /> : <Copy className="h-4 w-4" />}
                      </Button>
                      <pre className="bg-muted p-4 rounded-lg text-xs font-mono overflow-x-auto whitespace-pre-wrap">
                        {run.sqlite.explain}
                      </pre>
                    </div>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>
          )}

          {/* QAOA Details */}
          {run.qaoa && run.qaoa.supported && !run.qaoa.error && (
            <AccordionItem value="qaoa-details">
              <AccordionTrigger className="text-sm font-medium flex items-center gap-2">
                <Zap className="h-4 w-4" />
                QAOA Quantum Details
              </AccordionTrigger>
              <AccordionContent>
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Execution Time:</span>
                      <span className="ml-2">{run.qaoa.execTimeMs?.toFixed(2)}ms</span>
                    </div>
                    <div>
                      <span className="font-medium">Total Time:</span>
                      <span className="ml-2">{run.qaoa.totalTimeMs?.toFixed(2)}ms</span>
                    </div>
                    <div>
                      <span className="font-medium">Estimated Cost:</span>
                      <span className="ml-2">{run.qaoa.cost?.toLocaleString()}</span>
                    </div>
                    <div className="col-span-2">
                      <span className="font-medium">Join Order:</span>
                      <span className="ml-2 font-mono">{run.qaoa.joinOrder}</span>
                    </div>
                    {run.qaoa.treeStructure && (
                      <div className="col-span-2">
                        <span className="font-medium">Tree Structure:</span>
                        <span className="ml-2 font-mono">{run.qaoa.treeStructure}</span>
                      </div>
                    )}
                  </div>
                  
                  <div className="bg-blue-50 dark:bg-blue-950/20 p-3 rounded-lg">
                    <p className="text-xs text-blue-800 dark:text-blue-200">
                      <strong>Quantum Note:</strong> QAOA uses quantum approximate optimization to find 
                      optimal join orders. The total time includes quantum circuit execution time.
                    </p>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          )}
        </Accordion>
      </CardContent>
    </Card>
  )
}
