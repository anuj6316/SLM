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
        <div className="absolute top-5 left-1/2 w-full h-[2px] bg-zinc-100 dark:bg-white/5 -z-10">
          <div 
            className={cn(
              "h-full transition-all duration-500",
              isComplete ? "bg-emerald-500" : isActive ? "bg-violet-500 w-1/2" : "w-0"
            )} 
          />
        </div>
      )}

      <div className={cn(
        "w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all duration-300 z-10 bg-white dark:bg-[#18181b]",
        isActive ? "border-violet-500 text-violet-500 shadow-[0_0_0_4px_rgba(139,92,246,0.1)]" : 
        isComplete ? "border-emerald-500 text-emerald-500" : "border-zinc-200 dark:border-white/10 text-zinc-400"
      )}>
        <step.icon className="w-4 h-4" />
      </div>
      
      <div className="mt-3 text-center">
        <h3 className={cn(
          "text-xs font-bold uppercase tracking-wider mb-0.5",
          isActive ? "text-violet-600 dark:text-violet-400" : "text-zinc-500 dark:text-zinc-400"
        )}>
          {step.label}
        </h3>
        <p className="text-[10px] text-zinc-400 dark:text-zinc-500 hidden sm:block">{step.description}</p>
      </div>
    </div>
  );
};

const MetricCard = ({ metric }: { metric: Metric }) => (
  <div className="p-6 rounded-2xl bg-white dark:bg-[#18181b] border border-zinc-100 dark:border-white/5 shadow-sm hover:shadow-md transition-shadow flex flex-col justify-between h-40 relative overflow-hidden group">
    <div className="relative z-10">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">{metric.label}</h4>
        <div className={cn(
          "p-1.5 rounded-full bg-zinc-50 dark:bg-white/5 text-zinc-400 group-hover:text-violet-500 transition-colors",
          metric.label.includes('Quality') && "text-emerald-500 bg-emerald-500/10"
        )}>
          {metric.label.includes('Quality') ? <Activity className="w-3 h-3" /> : <Zap className="w-3 h-3" />}
        </div>
      </div>
      <div className="text-3xl font-bold text-zinc-900 dark:text-white tracking-tight">{metric.value}</div>
      <div className="text-[10px] text-zinc-400 mt-1 flex items-center gap-1">
        <span className="text-emerald-500 font-medium">+12%</span> from last week
      </div>
    </div>
    
    <div className="absolute bottom-0 left-0 right-0 h-16 opacity-30 group-hover:opacity-50 transition-opacity">
      <ResponsiveContainer width="100%" height="100%">
        {metric.isBar ? (
          <BarChart data={metric.trend.map((v, i) => ({ v, i }))}>
            <Bar dataKey="v" fill="#8b5cf6" radius={[2, 2, 0, 0]} />
          </BarChart>
        ) : (
          <AreaChart data={metric.trend.map((v, i) => ({ v, i }))}>
            <defs>
              <linearGradient id={`gradient-${metric.id}`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.5}/>
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <Area 
              type="monotone" 
              dataKey="v" 
              stroke="#8b5cf6" 
              strokeWidth={2} 
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
          <div className="bg-white dark:bg-[#18181b] rounded-3xl border border-zinc-100 dark:border-white/5 p-8 shadow-sm">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-lg font-bold text-zinc-900 dark:text-white">Quality Distribution</h3>
                <p className="text-sm text-zinc-500 dark:text-zinc-400">Weekly performance metrics</p>
              </div>
              <select className="bg-zinc-50 dark:bg-white/5 border-none text-xs rounded-lg px-3 py-2 text-zinc-600 dark:text-zinc-300 outline-none cursor-pointer">
                <option>This Week</option>
                <option>Last Week</option>
              </select>
            </div>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={qualityData} barSize={40}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" opacity={0.1} />
                  <XAxis 
                    dataKey="name" 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{ fill: '#A1A1AA', fontSize: 12 }} 
                    dy={10}
                  />
                  <YAxis 
                    axisLine={false} 
                    tickLine={false} 
                    tick={{ fill: '#A1A1AA', fontSize: 12 }} 
                  />
                  <Tooltip 
                    cursor={{ fill: 'transparent' }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        return (
                          <div className="bg-zinc-900 text-white text-xs py-2 px-3 rounded-lg shadow-xl">
                            <span className="font-bold">{payload[0].value}</span> Quality Score
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar dataKey="value" radius={[8, 8, 8, 8]}>
                    {qualityData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={index === 1 ? '#8b5cf6' : '#E4E4E7'} 
                        className="dark:fill-violet-500/20 dark:hover:fill-violet-500 transition-colors"
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Live Logs */}
          <div className="bg-[#09090b] rounded-3xl border border-zinc-800 overflow-hidden shadow-2xl flex flex-col h-[400px]">
            <div className="px-6 py-4 border-b border-zinc-800 bg-[#18181b] flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-[#FF5F56]" />
                  <div className="w-3 h-3 rounded-full bg-[#FFBD2E]" />
                  <div className="w-3 h-3 rounded-full bg-[#27C93F]" />
                </div>
                <span className="text-xs font-mono text-zinc-400 ml-2">process_monitor.exe</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="text-[10px] text-emerald-500 font-mono uppercase">Live</span>
              </div>
            </div>
            <div className="flex-1 p-6 font-mono text-xs leading-relaxed overflow-y-auto scrollbar-hide">
              {logs.map((log, i) => (
                <div key={log.id} className="mb-2 flex gap-3 opacity-0 animate-in fade-in slide-in-from-bottom-2 duration-300" style={{ animationDelay: `${i * 50}ms`, animationFillMode: 'forwards' }}>
                  <span className="text-zinc-600 select-none">{(i + 1).toString().padStart(3, '0')}</span>
                  <span className={cn(
                    "font-bold w-12",
                    log.type === 'INFO' ? "text-emerald-400" : "text-red-400"
                  )}>{log.type}</span>
                  {log.scope && <span className="text-violet-400 w-20">[{log.scope}]</span>}
                  <span className="text-zinc-300 flex-1">{log.message}</span>
                </div>
              ))}
              <div className="flex gap-3 mt-2">
                <span className="text-zinc-600 select-none">...</span>
                <span className="animate-pulse text-emerald-500">▋</span>
              </div>
            </div>
          </div>

        </div>

        {/* Right Column: Recent Datasets */}
        <div className="xl:col-span-1 bg-white dark:bg-[#18181b] rounded-3xl border border-zinc-100 dark:border-white/5 p-8 shadow-sm flex flex-col h-full">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-bold text-zinc-900 dark:text-white">Recent Datasets</h3>
            <button className="p-2 hover:bg-zinc-50 dark:hover:bg-white/5 rounded-full transition-colors">
              <MoreVertical className="w-5 h-5 text-zinc-400" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto -mx-4 px-4">
            <div className="space-y-3">
              {datasets.map((item) => (
                <div key={item.id} className="group p-4 rounded-2xl border border-zinc-100 dark:border-white/5 hover:border-violet-500/30 hover:bg-violet-50/50 dark:hover:bg-violet-500/5 transition-all cursor-pointer">
                  <div className="flex items-start justify-between mb-2">
                    <div className="p-2 rounded-lg bg-zinc-50 dark:bg-white/5 text-violet-500 group-hover:bg-violet-500 group-hover:text-white transition-colors">
                      {typeof item.id === 'number' && item.id % 2 === 0 ? <Database className="w-4 h-4" /> : <FileCode className="w-4 h-4" />}
                    </div>
                    <span className="text-[10px] font-medium text-zinc-400 bg-zinc-100 dark:bg-white/5 px-2 py-1 rounded-full">
                      {item.type}
                    </span>
                  </div>
                  <h4 className="text-sm font-semibold text-zinc-900 dark:text-white mb-1 truncate">{item.name}</h4>
                  <div className="flex items-center justify-between text-xs text-zinc-500 dark:text-zinc-400">
                    <span>{item.createdRelative}</span>
                    <ArrowRight className="w-3 h-3 opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 transition-all" />
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <button className="w-full mt-6 py-3 rounded-xl border border-zinc-200 dark:border-white/10 text-sm font-medium text-zinc-600 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors">
            View All Datasets
          </button>
        </div>

      </section>
    </div>
  );
}
