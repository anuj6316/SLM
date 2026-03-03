import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  Database, 
  Settings as SettingsIcon, 
  Bot, 
  Sliders, 
  Bell, 
  Play, 
  Loader2, 
  Search, 
  Moon, 
  Sun,
  Layers
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { api } from '@/services/api';
import { 
  SystemStatus, 
  Metric, 
  LogEntry, 
  QualityDataPoint, 
  Dataset
} from '@/types';

// Pages
import Dashboard from '@/pages/Dashboard';
import Scraper from '@/pages/Scraper';
import QAGen from '@/pages/QAGen';
import Config from '@/pages/Config';
import Datasets from '@/pages/Datasets';
import Settings from '@/pages/Settings';
import Signup from '@/pages/Signup';
import Login from '@/pages/Login';

// --- Components ---

const SidebarItem = ({ 
  icon: Icon, 
  label, 
  active = false,
  onClick
}: { 
  icon: React.ElementType, 
  label: string, 
  active?: boolean,
  onClick: () => void
}) => (
  <div 
    onClick={onClick}
    className={cn(
      "flex flex-col items-center justify-center py-4 px-2 cursor-pointer transition-colors group w-full relative",
      active ? "text-violet-500 dark:text-violet-400" : "text-zinc-500 dark:text-zinc-400 hover:text-violet-500 dark:hover:text-violet-400"
    )}
  >
    <Icon className={cn("w-6 h-6 mb-1 transition-transform group-hover:scale-110", active && "scale-110")} />
    <span className="text-[10px] font-medium text-center">{label}</span>
    {active && <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-violet-500 rounded-r-full" />}
  </div>
);

const StatusBadge = ({ label, status }: { label: string, status: 'online' | 'offline' }) => (
  <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-zinc-100 dark:bg-[#27272a] border border-zinc-200 dark:border-white/10">
    <div className={cn("w-2 h-2 rounded-full animate-pulse", status === 'online' ? "bg-emerald-500" : "bg-red-500")} />
    <span className="text-xs font-medium text-zinc-600 dark:text-zinc-300">
      <span className="text-zinc-400 dark:text-zinc-500 mr-1">{label}:</span>
      <span className={cn(status === 'online' ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400")}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    </span>
  </div>
);

type View = 'dashboard' | 'scraper' | 'qagen' | 'config' | 'datasets' | 'settings' | 'signup' | 'login';

export default function App() {
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [isLoading, setIsLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentView, setCurrentView] = useState<View>('login');
  const [hasNotifications, setHasNotifications] = useState(true);
  
  // State for API data
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [qualityData, setQualityData] = useState<QualityDataPoint[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);

  // Toggle dark mode class on html element
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  // Check for existing token
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      setIsAuthenticated(true);
      setCurrentView('dashboard');
    }
  }, []);

  // Fetch initial data
  useEffect(() => {
    if (isAuthenticated) {
      const fetchData = async () => {
        try {
          const [
            statusData,
            metricsData,
            logsData,
            qualityRes,
            datasetsRes
          ] = await Promise.all([
            api.getStatus(),
            api.getMetrics(),
            api.getLogs(),
            api.getQualityDistribution(),
            api.getRecentDatasets()
          ]);

          setStatus(statusData);
          setMetrics(metricsData.metrics);
          setLogs(logsData.logs);
          setQualityData(qualityRes.data);
          setDatasets(datasetsRes.datasets);
        } catch (error) {
          console.error('Failed to fetch dashboard data:', error);
        } finally {
          setIsLoading(false);
        }
      };

      fetchData();
    } else {
      setIsLoading(false);
    }
  }, [isAuthenticated]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    setIsAuthenticated(false);
    setCurrentView('login');
  };

  const handleRunPipeline = async () => {
    try {
      await api.runPipeline({ configId: 'default-v1' });
      // In a real app, you'd probably poll for status updates here
      const newStatus = await api.getStatus();
      setStatus(newStatus);
    } catch (error) {
      console.error('Failed to run pipeline:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-zinc-50 dark:bg-[#09090b]">
        <Loader2 className="w-8 h-8 animate-spin text-violet-500" />
      </div>
    );
  }

  // Auth Guard
  if (!isAuthenticated) {
    if (currentView === 'signup') {
      return <Signup onLoginClick={() => setCurrentView('login')} />;
    }
    return (
      <Login 
        onSignupClick={() => setCurrentView('signup')} 
        onLoginSuccess={() => {
          setIsAuthenticated(true);
          setCurrentView('dashboard');
          setIsLoading(true); // Ensure loading is triggered for dashboard data
        }}
      />
    );
  }

  // Loading Screen for Dashboard Data
  if (isLoading || !status) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-zinc-50 dark:bg-[#09090b]">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="w-10 h-10 animate-spin text-violet-500" />
          <p className="text-sm font-medium text-zinc-500 animate-pulse">Initializing Dashboard...</p>
        </div>
      </div>
    );
  }

  const renderContent = () => {
    switch (currentView) {
      case 'dashboard':
        return (
          <Dashboard 
            status={status} 
            metrics={metrics} 
            logs={logs} 
            qualityData={qualityData} 
            datasets={datasets} 
          />
        );
      case 'scraper':
        return <Scraper />;
      case 'qagen':
        return <QAGen />;
      case 'config':
        return (
          <Config 
            onStatusUpdate={(newStatus: SystemStatus) => setStatus(newStatus)} 
          />
        );
      case 'datasets':
        return <Datasets />;
      case 'settings':
        return <Settings />;
      default:
        return <Dashboard 
          status={status} 
          metrics={metrics} 
          logs={logs} 
          qualityData={qualityData} 
          datasets={datasets} 
        />;
    }
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-zinc-50 dark:bg-[#09090b]">
      {/* Sidebar */}
      <aside className="w-20 flex-shrink-0 border-r border-zinc-200 dark:border-white/10 bg-white dark:bg-[#18181b] flex flex-col items-center py-6 z-20">
        <div className="mb-8">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-violet-600 to-indigo-600 flex items-center justify-center shadow-xl shadow-violet-500/20 relative overflow-hidden group cursor-pointer transition-transform hover:scale-105 duration-300">
            <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
            <svg className="w-6 h-6 text-white drop-shadow-md" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
              <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
              <line x1="12" y1="22.08" x2="12" y2="12" />
            </svg>
          </div>
        </div>
        
        <nav className="flex-1 w-full space-y-2">
          <SidebarItem 
            icon={LayoutDashboard} 
            label="Dashboard" 
            active={currentView === 'dashboard'} 
            onClick={() => setCurrentView('dashboard')}
          />
          <SidebarItem 
            icon={Search} 
            label="Scraper" 
            active={currentView === 'scraper'} 
            onClick={() => setCurrentView('scraper')}
          />
          <SidebarItem 
            icon={Bot} 
            label="QA Gen" 
            active={currentView === 'qagen'} 
            onClick={() => setCurrentView('qagen')}
          />
          <SidebarItem 
            icon={Sliders} 
            label="Config" 
            active={currentView === 'config'} 
            onClick={() => setCurrentView('config')}
          />
          <SidebarItem 
            icon={Database} 
            label="Datasets" 
            active={currentView === 'datasets'} 
            onClick={() => setCurrentView('datasets')}
          />
          <SidebarItem 
            icon={SettingsIcon} 
            label="Settings" 
            active={currentView === 'settings'} 
            onClick={() => setCurrentView('settings')}
          />
        </nav>

        <div className="mt-auto pt-4 border-t border-zinc-200 dark:border-white/10 w-full flex flex-col items-center gap-4">
          <button 
            onClick={() => setIsDarkMode(!isDarkMode)}
            className="p-2 rounded-xl text-zinc-500 hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors"
          >
            {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
          <button 
            onClick={handleLogout}
            className="w-8 h-8 rounded-full bg-violet-500 flex items-center justify-center text-white font-bold text-xs ring-2 ring-white dark:ring-[#18181b] hover:scale-110 transition-transform"
            title="Log Out"
          >
            JD
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Header */}
        <header className="h-20 flex-shrink-0 px-8 flex items-center justify-between border-b border-zinc-200 dark:border-white/10 bg-white/80 dark:bg-[#09090b]/80 backdrop-blur-md z-10">
          <div>
            <h1 className="text-xl font-semibold text-zinc-900 dark:text-white">QA Data Pipeline</h1>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">Unstructured to QA</p>
          </div>

          <div className="flex items-center gap-4">
            <StatusBadge label="Jina" status={status?.services.jina || 'offline'} />
            <StatusBadge label="Groq" status={status?.services.groq || 'offline'} />
            
            <button 
              onClick={() => setHasNotifications(!hasNotifications)}
              className="p-2 text-zinc-500 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-white/5 rounded-xl transition-colors relative"
            >
              <Bell className="w-5 h-5" />
              {hasNotifications && (
                <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full border-2 border-white dark:border-[#09090b]" />
              )}
            </button>

            <button 
              onClick={handleRunPipeline}
              className="flex items-center gap-2 bg-violet-600 hover:bg-violet-700 text-white px-4 py-2 rounded-xl font-medium shadow-lg shadow-violet-500/20 transition-all active:scale-95"
            >
              <Play className="w-4 h-4 fill-current" />
              RUN PIPELINE
            </button>
          </div>
        </header>

        {/* Scrollable Content - Full Width */}
        <div className="flex-1 overflow-y-auto p-8 scrollbar-hide">
          <div className="w-full space-y-6">
            {renderContent()}
          </div>
        </div>
      </main>
    </div>
  );
}
