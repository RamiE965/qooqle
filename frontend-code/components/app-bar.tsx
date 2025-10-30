"use client"

import { useTheme } from "./theme-provider"
import { Moon, Sun } from "lucide-react"
import { Button } from "./ui/button"

export function AppBar() {
  const { theme, toggleTheme } = useTheme()

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/80">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex gap-1">
              <div className="w-2 h-2 rounded-full bg-[#4285F4]" />
              <div className="w-2 h-2 rounded-full bg-[#EA4335]" />
              <div className="w-2 h-2 rounded-full bg-[#FBBC04]" />
              <div className="w-2 h-2 rounded-full bg-[#34A853]" />
            </div>
            <div>
              <h1 className="text-xl font-medium text-foreground">Qooqle</h1>
              <p className="text-xs text-muted-foreground">Query Join Reordering Using Quantum Simulation</p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={toggleTheme} className="rounded-full">
            {theme === "light" ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
            <span className="sr-only">Toggle theme</span>
          </Button>
        </div>
      </div>
    </header>
  )
}
