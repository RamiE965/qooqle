// import { Card, CardContent, CardHeader, CardTitle } from "./ui/card"
// import { AlertCircle } from "lucide-react"

// interface EngineData {
//   timeMs?: number
//   execTimeMs?: number
//   totalTimeMs?: number
//   cost?: number
//   joinOrder?: string
//   error?: string
//   supported?: boolean
// }

// const ENGINE_COLORS = {
//   DuckDB: "bg-[#4285F4] text-white", // Google Blue
//   SQLite: "bg-[#EA4335] text-white", // Google Red
//   QAOA: "bg-[#34A853] text-white", // Google Green
// }

// export function StatCard({ title, data }: { title: string; data?: EngineData }) {
//   const isQaoa = title === "QAOA"
//   const hasError = data?.error || (isQaoa && data?.supported === false)
//   const joinOrderArray = data?.joinOrder?.split(" → ") || []
//   const engineColor = ENGINE_COLORS[title as keyof typeof ENGINE_COLORS]

//   return (
//     <Card className={`rounded-xl overflow-hidden shadow-sm ${hasError ? "opacity-60" : ""}`}>
//       {/* remove rounded-t-* here */}
//       <CardHeader className={`pb-3 ${engineColor} border-b border-black/10`}>
//         <CardTitle className="text-sm font-medium">{title}</CardTitle>
//       </CardHeader>

//       <CardContent className="space-y-3 pt-4">
//         {hasError ? (
//           <div className="flex items-start gap-2 text-sm text-muted-foreground">
//             <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
//             <span>{data?.error || "Not supported"}</span>
//           </div>
//         ) : (
//           <>
//             {/* Join Order */}
//             {joinOrderArray.length > 0 && (
//               <div>
//                 <p className="text-xs text-muted-foreground mb-2">Join Order</p>
//                 <div className="flex flex-wrap gap-1">
//                   {joinOrderArray.map((alias, i) => (
//                     <span key={i} className="inline-flex items-center gap-1">
//                       {/* chips can keep the engine color */}
//                       <span className={`px-2 py-1 ${engineColor} rounded text-xs font-mono`}>
//                         {alias}
//                       </span>
//                       {i < joinOrderArray.length - 1 && <span className="text-muted-foreground">→</span>}
//                     </span>
//                   ))}
//                 </div>
//               </div>
//             )}

//             {/* Time */}
//             <div>
//               <p className="text-xs text-muted-foreground">Time (ms)</p>
//               <p className="text-2xl font-medium">
//                 {isQaoa ? data?.totalTimeMs?.toFixed(2) || "—" : data?.timeMs?.toFixed(2) || "—"}
//               </p>
//               {isQaoa && data?.execTimeMs !== undefined && (
//                 <p className="text-xs text-muted-foreground">Exec: {data.execTimeMs.toFixed(2)} ms</p>
//               )}
//             </div>

//             {/* Cost */}
//             {data?.cost !== undefined && (
//               <div>
//                 <p className="text-xs text-muted-foreground">Cost</p>
//                 <p className="text-lg font-medium">{data.cost.toLocaleString()}</p>
//               </div>
//             )}
//           </>
//         )}
//       </CardContent>
//     </Card>
//   )
// }

import { Card } from "./ui/card"
import { AlertCircle } from "lucide-react"

interface EngineData {
  timeMs?: number
  execTimeMs?: number
  totalTimeMs?: number
  cost?: number
  joinOrder?: string
  error?: string
  supported?: boolean
}

const ENGINE_COLORS = {
  DuckDB: "bg-[#4285F4] text-white", // Google Blue
  SQLite: "bg-[#EA4335] text-white", // Google Red
  QAOA: "bg-[#34A853] text-white",   // Google Green
}

export function StatCard({ title, data }: { title: string; data?: EngineData }) {
  const isQaoa = title === "QAOA"
  const hasError = Boolean(data?.error) || (isQaoa && data?.supported === false)
  const joinOrderArray = data?.joinOrder?.split(" → ") ?? []
  const engineColor = ENGINE_COLORS[title as keyof typeof ENGINE_COLORS]

  return (
    <Card className={`rounded-2xl overflow-hidden border shadow-sm ${hasError ? "opacity-60" : ""}`}>
      {/* FULL-BLEED TOP BAR — no rounding here; card handles it */}
      <div className={`h-12 ${engineColor} flex items-center px-4 text-sm font-medium`}>
        {title}
      </div>

      {/* smaller, tighter content */}
      <div className="p-4 space-y-3">
        {hasError ? (
          <div className="flex items-start gap-2 text-sm text-muted-foreground">
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
            <span>{data?.error || "Not supported"}</span>
          </div>
        ) : (
          <>
            {/* Join Order */}
            {joinOrderArray.length > 0 && (
              <div>
                <p className="text-xs text-muted-foreground mb-1.5">Join Order</p>
                <div className="flex flex-wrap gap-1">
                  {joinOrderArray.map((alias, i) => (
                    <span key={`${alias}-${i}`} className="inline-flex items-center gap-1">
                      <span className={`px-2 py-0.5 rounded text-xs font-mono ${engineColor}`}>
                        {alias}
                      </span>
                      {i < joinOrderArray.length - 1 && (
                        <span className="text-muted-foreground">→</span>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Time */}
            <div>
              <p className="text-xs text-muted-foreground">Time (ms)</p>
              <p className="text-3xl font-semibold leading-tight">
                {isQaoa
                  ? (data?.totalTimeMs != null ? data.totalTimeMs.toFixed(2) : "—")
                  : (data?.timeMs != null ? data.timeMs.toFixed(2) : "—")}
              </p>
              {isQaoa && data?.execTimeMs != null && (
                <p className="text-xs text-muted-foreground mt-0.5">
                  Exec: {data.execTimeMs.toFixed(2)} ms
                </p>
              )}
            </div>

            {/* Cost */}
            {data?.cost != null && (
              <div>
                <p className="text-xs text-muted-foreground">Cost</p>
                <p className="text-xl font-medium">{data.cost.toLocaleString()}</p>
              </div>
            )}
          </>
        )}
      </div>
    </Card>
  )
}
