import React from 'react';
import { 
  FileText, 
  Search, 
  Sliders, 
  Bot, 
  Database, 
  MoreVertical,
  FileCode,
  ArrowRight,
  Activity,
  Zap,
  Clock
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Cell,
  AreaChart,
  Area
} from 'recharts';
import { motion } from 'motion/react';
import { cn } from '@/lib/utils';
import { 
  SystemStatus, 
  Metric, 
  LogEntry, 
  QualityDataPoint, 
  Dataset,
  PipelineStep as PipelineStepType
} from '@/types';

// --- Components ---

type PipelineStepUI = PipelineStepType & {
  label: string;
  icon: React.ElementType;
  description: string;
};

const STEP_CONFIG: Record<string, { label: string; icon: React.ElementType; description: string }> = {
  input: { label: 'Ingestion', icon: FileText, description: 'Raw data processing' },
  scraping: { label: 'Extraction', icon: Search, description: 'Web & document scraping' },
  refinement: { label: 'Cleaning', icon: Sliders, description: 'Data normalization' },
  aigen: { label: 'Synthesis', icon: Bot, description: 'LLM generation' },
  output: { label: 'Delivery', icon: Database, description: 'Final QA export' },
};

const PipelineStep = ({ step, index, total }: { step: PipelineStepUI; index: number; total: number }) => {
  const isActive = step.status === 'active';
  const isComplete = step.status === 'complete';
  
  return (
    <div className="relative flex-1 flex flex-col items-center group">
      {/* Connector Line */}
      {index < total - 1 && (
        <div className="absolute top-5 left-1/2 w-full h-[1px] bg-zinc-200 dark:bg-white/5 -z-10">
          <div 
            className={cn(
              "h-full transition-all duration-1000",
              isComplete ? "bg-[var(--brand-success)]" : isActive ? "bg-[var(--brand-accent)] w-1/2" : "w-0"
            )} 
          />
        </div>
      )}

      <div className={cn(
        "w-10 h-10 rounded-xl flex items-center justify-center border transition-all duration-500 z-10",
        isActive ? "bg-[var(--brand-accent)] border-[var(--brand-accent)] text-white shadow-lg shadow-violet-500/30 scale-110" : 
        isComplete ? "bg-[var(--brand-success)]/10 border-[var(--brand-success)] text-[var(--brand-success)]" : 
        "bg-[var(--bg-surface)] border-zinc-200 dark:border-white/5 text-zinc-400"
      )}>
        <step.icon className={cn("w-4 h-4", isActive && "animate-pulse")} />
      </div>
      
      <div className="mt-4 text-center">
        <h3 className={cn(
          "text-[10px] font-bold uppercase tracking-[0.15em] mb-0.5 transition-colors",
          isActive ? "text-[var(--brand-accent)]" : "text-zinc-500 dark:text-zinc-400"
        )}>
          {step.label}
        </h3>
        <p className="text-[9px] text-zinc-400 font-medium hidden lg:block opacity-60">{step.description}</p>
      </div>
    </div>
  );
};

const MetricCard = ({ metric }: { metric: Metric }) => (
  <div className="p-6 rounded-3xl bg-[var(--bg-surface)] border border-zinc-200 dark:border-white/5 shadow-sm hover:border-[var(--brand-accent)]/20 transition-all flex flex-col justify-between h-40 relative overflow-hidden group">
    <div className="relative z-10">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">{metric.label}</h4>
        <div className={cn(
          "p-1.5 rounded-lg bg-zinc-50 dark:bg-white/5 text-zinc-400 group-hover:text-[var(--brand-accent)] transition-colors",
          metric.label.includes('Quality') && "text-[var(--brand-success)] bg-[var(--brand-success)]/10"
        )}>
          {metric.label.includes('Quality') ? <Activity className="w-3.5 h-3.5" /> : <Zap className="w-3.5 h-3.5" />}
        </div>
      </div>
      <div className="text-3xl font-bold text-[var(--text-main)] tracking-tight font-display">{metric.value}</div>
      <div className="text-[10px] mt-1 font-bold flex items-center gap-1">
        <span className="text-[var(--brand-success)]">↑ 12%</span>
        <span className="text-zinc-400 font-medium">vs last run</span>
      </div>
    </div>
    
    <div className="absolute bottom-0 left-0 right-0 h-14 opacity-20 group-hover:opacity-40 transition-opacity">
      <ResponsiveContainer width="100%" height="100%">
        {metric.isBar ? (
          <BarChart data={metric.trend.map((v, i) => ({ v, i }))}>
            <Bar dataKey="v" fill="var(--brand-accent)" radius={[2, 2, 0, 0]} />
          </BarChart>
        ) : (
          <AreaChart data={metric.trend.map((v, i) => ({ v, i }))}>
            <defs>
              <linearGradient id={`gradient-${metric.id}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--brand-accent)" stopOpacity={0.6}/>
                <stop offset="100%" stopColor="var(--brand-accent)" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <Area 
              type="monotone" 
              dataKey="v" 
              stroke="var(--brand-accent)" 
              strokeWidth={1.5} 
              fill={`url(#gradient-${metric.id})`} 
            />
          </AreaChart>
        )}
      </ResponsiveContainer>
    </div>
  </div>
);

interface DashboardProps {
  status: SystemStatus | null;
  metrics: Metric[];
  logs: LogEntry[];
  qualityData: QualityDataPoint[];
  datasets: Dataset[];
}

export default function Dashboard({ status, metrics, logs, qualityData, datasets }: DashboardProps) {
  const pipelineSteps: PipelineStepUI[] = status?.pipeline.steps.map(step => ({
    ...step,
    label: STEP_CONFIG[step.id]?.label || step.id,
    description: STEP_CONFIG[step.id]?.description || '',
    icon: STEP_CONFIG[step.id]?.icon || FileText
  })) || [];

  return (
    <div className="space-y-8 max-w-[1600px] mx-auto">
      
      {/* Top Section: Pipeline Flow */}
      <section className="bg-white dark:bg-[#18181b] rounded-3xl border border-zinc-100 dark:border-white/5 p-8 shadow-sm">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-lg font-bold text-zinc-900 dark:text-white">Pipeline Status</h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">Real-time processing overview</p>
          </div>
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-violet-50 dark:bg-violet-500/10 text-violet-600 dark:text-violet-400 text-xs font-medium">
            <Clock className="w-3 h-3" />
            <span>Est. completion: 12m 30s</span>
          </div>
        </div>
        <div className="flex justify-between items-start px-4">
          {pipelineSteps.map((step, i) => (
            <PipelineStep key={step.id} step={step} index={i} total={pipelineSteps.length} />
          ))}
        </div>
      </section>

      {/* Metrics Grid */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-6">
        {metrics.map((metric) => (
          <MetricCard key={metric.id} metric={metric} />
        ))}
      </section>

      {/* Main Content Grid */}
      <section className="grid grid-cols-1 xl:grid-cols-3 gap-8">
        
        {/* Left Column: Charts & Logs */}
        <div className="xl:col-span-2 space-y-8">
          
          {/* Quality Chart */}
          <div className="bg-[var(--bg-surface)] rounded-3xl border border-zinc-200 dark:border-white/5 p-8 shadow-sm relative overflow-hidden">
            <div className="flex items-center justify-between mb-8">
              <div>
                <h3 className="text-lg font-bold text-[var(--text-main)] font-display tracking-tight">Quality Index</h3>
                <p className="text-[10px] text-zinc-400 font-bold uppercase tracking-[0.2em] mt-1">Algorithm Performance Distribution</p>
              </div>
              <div className="flex gap-2">
                <button className="px-3 py-1.5 rounded-lg bg-zinc-100 dark:bg-white/5 text-[10px] font-bold text-zinc-500 uppercase tracking-wider hover:text-[var(--brand-accent)] transition-colors">7D</button>
                <button className="px-3 py-1.5 rounded-lg bg-[var(--brand-accent)] text-white text-[10px] font-bold uppercase tracking-wider shadow-lg shadow-violet-500/20">30D</button>
              </div>
            </div>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={qualityData} barSize={32}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="currentColor" className="text-zinc-200 dark:text-zinc-800" opacity={0.5} />
                  <XAxis 
                    dataKey="name" 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{ fill: '#71717a', fontSize: 10, fontWeight: 700 }} 
                    dy={10}
                  />
                  <YAxis 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{ fill: '#71717a', fontSize: 10, fontWeight: 700 }} 
                  />
                  <Tooltip 
                    cursor={{ fill: 'rgba(0,0,0,0.02)', radius: 8 }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="bg-zinc-900 border border-white/10 text-white text-[10px] font-bold py-2.5 px-4 rounded-2xl shadow-2xl backdrop-blur-xl">
                            <div className="text-zinc-500 uppercase tracking-widest mb-1 text-[8px]">Index Score</div>
                            <div className="text-lg text-[var(--brand-accent)] font-display">{payload[0].value}%</div>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar dataKey="value" radius={[6, 6, 6, 6]}>
                    {qualityData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={index === qualityData.length - 1 ? 'var(--brand-accent)' : 'rgba(113, 113, 122, 0.15)'} 
                        className="transition-all duration-500 hover:opacity-80"
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Live Logs - Professional Terminal */}
          <div className="bg-white dark:bg-[#0d0d12] rounded-3xl border border-zinc-200 dark:border-zinc-800/50 overflow-hidden shadow-sm dark:shadow-2xl flex flex-col h-[400px] group transition-all duration-500">
            <div className="px-6 py-4 border-b border-zinc-100 dark:border-zinc-800/50 bg-zinc-50/50 dark:bg-[#12121a] flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full bg-red-400 dark:bg-red-500/80" />
                  <div className="w-2.5 h-2.5 rounded-full bg-amber-400 dark:bg-amber-500/80" />
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 dark:bg-emerald-500/80" />
                </div>
                <div className="h-4 w-[1px] bg-zinc-200 dark:bg-zinc-800 mx-2" />
                <span className="text-[10px] font-bold text-zinc-400 dark:text-zinc-500 tracking-[0.2em] uppercase font-mono">Kernel.Mainframe.Monitor</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--brand-accent)] animate-pulse shadow-[0_0_8px_var(--brand-accent)]" />
                <span className="text-[10px] text-[var(--brand-accent)] font-bold uppercase tracking-widest font-mono">Live</span>
              </div>
            </div>
            <div className="flex-1 p-6 font-mono text-[11px] leading-relaxed overflow-y-auto custom-scrollbar bg-zinc-50/30 dark:bg-[#0d0d12]">
              {logs.map((log, i) => (
                <div key={log.id} className="mb-2 flex gap-4 opacity-0 animate-in fade-in slide-in-from-bottom-1 duration-500" style={{ animationDelay: `${i * 80}ms`, animationFillMode: 'forwards' }}>
                  <span className="text-zinc-300 dark:text-zinc-700 select-none w-6 text-right">{ (i + 1).toString().padStart(2, '0') }</span>
                  <span className={cn(
                    "font-bold w-10",
                    log.type === 'INFO' ? "text-emerald-600 dark:text-emerald-500/90" : "text-red-600 dark:text-red-500/90"
                  )}>{log.type}</span>
                  {log.scope && <span className="text-[var(--brand-accent)]/80 w-16">[{log.scope}]</span>}
                  <span className="text-zinc-600 dark:text-zinc-400 flex-1">{log.message}</span>
                </div>
              ))}
              <div className="flex gap-3 mt-4">
                <span className="text-zinc-200 dark:text-zinc-800 select-none w-6 text-right">--</span>
                <span className="animate-pulse text-[var(--brand-accent)] font-bold">_</span>
              </div>
            </div>
          </div>

        </div>

        {/* Right Column: Recent Datasets */}
        <div className="xl:col-span-1 bg-[var(--bg-surface)] rounded-3xl border border-zinc-200 dark:border-white/5 p-8 shadow-sm flex flex-col h-full">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h3 className="text-lg font-bold text-[var(--text-main)]">Recent Datasets</h3>
              <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-widest mt-1">Managed Repositories</p>
            </div>
            <button className="p-2 hover:bg-zinc-100 dark:hover:bg-white/5 rounded-xl transition-colors">
              <MoreVertical className="w-4 h-4 text-zinc-400" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto custom-scrollbar -mx-2 px-2">
            <div className="space-y-3">
              {datasets.map((item) => (
                <div key={item.id} className="group p-4 rounded-2xl border border-zinc-100 dark:border-white/5 hover:border-[var(--brand-accent)]/30 bg-zinc-50/50 dark:bg-white/[0.02] hover:bg-[var(--brand-accent)]/[0.02] transition-all cursor-pointer">
                  <div className="flex items-start justify-between mb-3">
                    <div className="p-2 rounded-xl bg-white dark:bg-white/5 text-zinc-400 group-hover:text-[var(--brand-accent)] group-hover:bg-[var(--brand-accent)]/10 transition-all">
                      {typeof item.id === 'number' && item.id % 2 === 0 ? <Database className="w-4 h-4" /> : <FileCode className="w-4 h-4" />}
                    </div>
                    <span className="text-[9px] font-bold text-zinc-500 dark:text-zinc-400 bg-white dark:bg-white/5 px-2 py-1 rounded-lg border border-zinc-100 dark:border-white/5">
                      {item.type}
                    </span>
                  </div>
                  <h4 className="text-sm font-bold text-[var(--text-main)] mb-1 truncate">{item.name}</h4>
                  <div className="flex items-center justify-between text-[10px] text-zinc-500 font-medium">
                    <span>{item.createdRelative}</span>
                    <ArrowRight className="w-3.5 h-3.5 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all text-[var(--brand-accent)]" />
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <button className="w-full mt-8 py-3 rounded-xl bg-zinc-50 dark:bg-white/5 border border-zinc-200 dark:border-white/10 text-[10px] font-bold text-zinc-600 dark:text-zinc-400 uppercase tracking-[0.2em] hover:bg-zinc-100 dark:hover:bg-white/10 transition-all">
            View All Datasets
          </button>
        </div>

      </section>
    </div>
  );
}
