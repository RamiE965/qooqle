"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
import { Button } from "./ui/button"
import { Textarea } from "./ui/textarea"
import { Checkbox } from "./ui/checkbox"
import { Label } from "./ui/label"
import { Upload, Database, Play } from "lucide-react"
import type { RunRequest } from "@/types/api"

const SAMPLE_QUERIES = {
  "3-table": `SELECT c.c_name, SUM(l.l_extendedprice) AS total_revenue
FROM lineitem l 
JOIN orders o ON l.l_orderkey = o.o_orderkey 
JOIN customer c ON o.o_custkey = c.c_custkey
GROUP BY c.c_name 
ORDER BY total_revenue DESC 
LIMIT 10`,
  
  "4-table": `SELECT n.n_name, c.c_name, SUM(l.l_extendedprice) AS total_revenue
FROM lineitem l 
JOIN orders o ON l.l_orderkey = o.o_orderkey 
JOIN customer c ON o.o_custkey = c.c_custkey 
JOIN nation n ON c.c_nationkey = n.n_nationkey
WHERE n.n_name = 'NATION_5'
GROUP BY n.n_name, c.c_name 
ORDER BY total_revenue DESC 
LIMIT 10`,
  
  "5-table": `SELECT r.r_name, n.n_name, c.c_name, SUM(l.l_extendedprice) AS total_revenue
FROM lineitem l 
JOIN orders o ON l.l_orderkey = o.o_orderkey 
JOIN customer c ON o.o_custkey = c.c_custkey 
JOIN nation n ON c.c_nationkey = n.n_nationkey 
JOIN region r ON n.n_regionkey = r.r_regionkey
WHERE r.r_name = 'REGION_2'
GROUP BY r.r_name, n.n_name, c.c_name 
ORDER BY total_revenue DESC 
LIMIT 10`
}

export function ConfigForm({ onRun, isLoading }: { onRun: (config: RunRequest) => void; isLoading: boolean }) {
  const [sqlQuery, setSqlQuery] = useState(SAMPLE_QUERIES["3-table"])
  const [databaseFile, setDatabaseFile] = useState<File | null>(null)
  const [engines, setEngines] = useState({
    duckdb: true,
    sqlite: true,
    qaoa: true,
  })

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setDatabaseFile(file)
    }
  }

  const handleRun = () => {
    if (!sqlQuery.trim()) return

    const config: RunRequest = {
      sqlQuery: sqlQuery.trim(),
      databaseFile: databaseFile || undefined,
      engines,
    }

    onRun(config)
  }

  const loadSampleQuery = (queryType: keyof typeof SAMPLE_QUERIES) => {
    setSqlQuery(SAMPLE_QUERIES[queryType])
  }

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle className="text-lg font-medium flex items-center gap-2">
          <Database className="h-5 w-5" />
          Query Optimizer Configuration
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* SQL Query Input */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="text-sm font-medium">SQL Query</Label>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => loadSampleQuery("3-table")}
                className="h-8 text-xs"
              >
                3 Tables
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => loadSampleQuery("4-table")}
                className="h-8 text-xs"
              >
                4 Tables
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => loadSampleQuery("5-table")}
                className="h-8 text-xs"
              >
                5 Tables
              </Button>
            </div>
          </div>
          <Textarea
            placeholder="Enter your SQL query here..."
            value={sqlQuery}
            onChange={(e) => setSqlQuery(e.target.value)}
            className="min-h-[200px] font-mono text-sm"
          />
        </div>

        {/* Database File Upload */}
        <div className="space-y-3">
          <Label className="text-sm font-medium flex items-center gap-2">
            <Upload className="h-4 w-4" />
            Database File (Optional)
          </Label>
          <div className="flex items-center gap-4">
            <input
              type="file"
              accept=".db,.sqlite,.sqlite3,.duckdb"
              onChange={handleFileUpload}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            {databaseFile && (
              <span className="text-sm text-green-600">
                ✓ {databaseFile.name}
              </span>
            )}
          </div>
          <p className="text-xs text-gray-500">
            Upload a database file to use your own data. If not provided, we'll use the built-in TPC-H schema.
          </p>
        </div>

        {/* Engine Selection */}
        <div className="space-y-3">
          <Label className="text-sm font-medium">Optimization Engines</Label>
          <div className="flex gap-6">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="duckdb"
                checked={engines.duckdb}
                onCheckedChange={(checked) => setEngines({ ...engines, duckdb: !!checked })}
              />
              <Label htmlFor="duckdb" className="text-sm font-normal cursor-pointer">
                DuckDB (Classical)
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="sqlite"
                checked={engines.sqlite}
                onCheckedChange={(checked) => setEngines({ ...engines, sqlite: !!checked })}
              />
              <Label htmlFor="sqlite" className="text-sm font-normal cursor-pointer">
                SQLite (Classical)
              </Label>
            </div>
            <div className="flex items-center space-x-2">
              <Checkbox
                id="qaoa"
                checked={engines.qaoa}
                onCheckedChange={(checked) => setEngines({ ...engines, qaoa: !!checked })}
              />
              <Label htmlFor="qaoa" className="text-sm font-normal cursor-pointer">
                QAOA (Quantum)
              </Label>
            </div>
          </div>
        </div>

        {/* Progress Warning */}
        {sqlQuery.toLowerCase().includes('region') && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <div className="flex items-center gap-2 text-yellow-800">
              <div className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium">5-Table Optimization Warning</span>
            </div>
            <p className="text-sm text-yellow-700 mt-1">
              This query involves 5 tables and may take up to 10 minutes to complete due to the complexity of quantum optimization.
            </p>
          </div>
        )}

        {/* Run Button */}
        <div className="flex justify-center pt-4">
          <Button 
            onClick={handleRun} 
            disabled={isLoading || !sqlQuery.trim()} 
            className="px-8 py-2"
            size="lg"
          >
            <Play className="h-4 w-4 mr-2" />
            {isLoading ? (
              <span className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                Running Optimization...
              </span>
            ) : (
              "Run Optimization"
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
