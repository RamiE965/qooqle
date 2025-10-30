export interface RunRequest {
  sqlQuery: string
  databaseFile?: File
  engines: {
    duckdb: boolean
    sqlite: boolean
    qaoa: boolean
  }
}

export interface EngineResult {
  timeMs?: number
  execTimeMs?: number
  totalTimeMs?: number
  cost?: number
  joinOrder?: string
  treeStructure?: string
  explain?: string
  error?: string
  supported?: boolean
}

export interface RunResult {
  runs: Array<{
    runIndex: number
    duckdb?: EngineResult
    sqlite?: EngineResult
    qaoa?: EngineResult & { supported: boolean }
    sql?: {
      qaoaForced?: string
    }
  }>
}
