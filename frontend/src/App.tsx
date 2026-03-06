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
  Layers,
  Key,
  ChevronDown,
  Save,
  RefreshCw,
  Eye,
  EyeOff,
  Check,
  AlertCircle,
  CheckCircle2,
  XCircle,
  Info,
  Trash2,
  MoreVertical,
  X,
  Palette
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

type Theme = 'light' | 'dark';

const ThemeSelector = ({ 
  currentTheme, 
  onThemeChange 
}: { 
  currentTheme: Theme, 
  onThemeChange: (theme: Theme) => void 
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const themes: { id: Theme, label: string, icon: any, colors: string[] }[] = [
    { id: 'light', label: 'Classic Light', icon: Sun, colors: ['bg-white', 'bg-zinc-100', 'bg-violet-600'] },
    { id: 'dark', label: 'Classic Dark', icon: Moon, colors: ['bg-zinc-950', 'bg-zinc-900', 'bg-violet-500'] },
  ];

  return (
    <div className="relative w-full flex justify-center" ref={containerRef}>
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 rounded-xl text-zinc-500 hover:bg-zinc-100 dark:hover:bg-white/5 transition-colors relative"
        title="Change Theme"
      >
        <Palette className="w-5 h-5" />
      </button>

      {isOpen && (
        <div className="absolute bottom-full left-14 mb-2 w-48 bg-white dark:bg-[#18181b] rounded-2xl border border-zinc-200 dark:border-white/10 shadow-2xl z-50 p-2 animate-in fade-in slide-in-from-left-2 duration-200">
          <div className="px-3 py-2 border-b border-zinc-100 dark:border-white/5 mb-1">
            <span className="text-[10px] font-bold text-zinc-400 uppercase tracking-widest">Select Theme</span>
          </div>
          {themes.map((t) => (
            <button
              key={t.id}
              onClick={() => {
                onThemeChange(t.id);
                setIsOpen(false);
              }}
              className={cn(
                "w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-medium transition-colors",
                currentTheme === t.id 
                  ? "bg-violet-50 dark:bg-violet-500/10 text-violet-600 dark:text-violet-400" 
                  : "text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-white/5"
              )}
            >
              <div className="flex items-center gap-2">
                <t.icon className="w-3.5 h-3.5" />
                {t.label}
              </div>
              <div className="flex gap-0.5">
                {t.colors.map((c, i) => (
                  <div key={i} className={cn("w-1.5 h-1.5 rounded-full", c)} />
                ))}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

interface Notification {
  id: string;
  type: 'success' | 'error' | 'info' | 'warning';
  title: string;
  message: string;
  timestamp: Date;
  read: boolean;
}

const NotificationDrawer = ({ 
  notifications, 
  onMarkRead, 
  onClearAll,
  onClose 
}: { 
  notifications: Notification[], 
  onMarkRead: (id: string) => void,
  onClearAll: () => void,
  onClose: () => void
}) => {
  return (
    <div className="absolute top-full mt-2 right-0 w-80 sm:w-96 bg-white dark:bg-[#18181b] rounded-3xl border border-zinc-200 dark:border-white/10 shadow-2xl z-50 overflow-hidden animate-in fade-in zoom-in-95 duration-200">
      <div className="p-4 border-b border-zinc-100 dark:border-white/5 flex items-center justify-between bg-zinc-50/50 dark:bg-white/5">
        <div className="flex items-center gap-2">
          <Bell className="w-4 h-4 text-violet-500" />
          <h3 className="font-bold text-sm text-zinc-900 dark:text-white">Notifications</h3>
          {notifications.filter(n => !n.read).length > 0 && (
            <span className="bg-violet-500 text-white text-[10px] px-1.5 py-0.5 rounded-full">
              {notifications.filter(n => !n.read).length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button 
            onClick={onClearAll}
            className="text-[10px] font-bold text-zinc-400 hover:text-red-500 transition-colors uppercase tracking-wider"
          >
            Clear All
          </button>
          <button onClick={onClose} className="p-1 hover:bg-zinc-200 dark:hover:bg-white/10 rounded-lg transition-colors">
            <X className="w-4 h-4 text-zinc-400" />
          </button>
        </div>
      </div>

      <div className="max-h-[400px] overflow-y-auto custom-scrollbar">
        {notifications.length === 0 ? (
          <div className="p-12 text-center">
            <div className="w-12 h-12 bg-zinc-100 dark:bg-white/5 rounded-2xl flex items-center justify-center mx-auto mb-3">
              <Bell className="w-6 h-6 text-zinc-300" />
            </div>
            <p className="text-sm font-medium text-zinc-500 dark:text-zinc-400">All caught up!</p>
            <p className="text-xs text-zinc-400 mt-1">No new notifications</p>
          </div>
        ) : (
          <div className="divide-y divide-zinc-100 dark:divide-white/5">
            {notifications.map((n) => (
              <div 
                key={n.id} 
                onClick={() => onMarkRead(n.id)}
                className={cn(
                  "p-4 hover:bg-zinc-50 dark:hover:bg-white/[0.02] transition-colors cursor-pointer group relative",
                  !n.read && "bg-violet-500/[0.02] dark:bg-violet-500/[0.03]"
                )}
              >
                {!n.read && <div className="absolute left-0 top-0 bottom-0 w-1 bg-violet-500" />}
                <div className="flex gap-3">
                  <div className={cn(
                    "w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0",
                    n.type === 'success' && "bg-emerald-100 dark:bg-emerald-500/10 text-emerald-600",
                    n.type === 'error' && "bg-red-100 dark:bg-red-500/10 text-red-600",
                    n.type === 'info' && "bg-blue-100 dark:bg-blue-500/10 text-blue-600",
                    n.type === 'warning' && "bg-amber-100 dark:bg-amber-500/10 text-amber-600"
                  )}>
                    {n.type === 'success' && <CheckCircle2 className="w-4 h-4" />}
                    {n.type === 'error' && <XCircle className="w-4 h-4" />}
                    {n.type === 'info' && <Info className="w-4 h-4" />}
                    {n.type === 'warning' && <AlertCircle className="w-4 h-4" />}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between items-start mb-0.5">
                      <h4 className="text-sm font-bold text-zinc-900 dark:text-white truncate pr-4">{n.title}</h4>
                      <span className="text-[10px] text-zinc-400 font-medium whitespace-nowrap">
                        {new Date(n.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                    <p className="text-xs text-zinc-500 dark:text-zinc-400 line-clamp-2 leading-relaxed">{n.message}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      
      {notifications.length > 0 && (
        <div className="p-3 bg-zinc-50/50 dark:bg-white/5 border-t border-zinc-100 dark:border-white/5 text-center">
          <button className="text-xs font-bold text-violet-600 dark:text-violet-400 hover:underline">
            View All Activity
          </button>
        </div>
      )}
    </div>
  );
};

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

const StatusBadgeDropdown = ({ 
  label, 
  status, 
  user, 
  type,
  onUpdate 
}: { 
  label: string, 
  status: 'online' | 'offline',
  user: any,
  type: 'jina_api_key' | 'groq_api_key',
  onUpdate: (updatedUser: any) => void
}) => {
  const [isOpen, setIsOpen] = React.useState(false);
  const [key, setKey] = React.useState(user?.[type] || '');
  const [showKey, setShowKey] = React.useState(false);
  const [isSaving, setIsSaving] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [showCheck, setShowCheck] = React.useState(false);

  const dropdownRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Update local key if user object changes
  React.useEffect(() => {
    setKey(user?.[type] || '');
  }, [user, type]);

  const handleSave = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsSaving(true);
    setError(null);
    try {
      const updatedUser = await api.updateKeys({ [type]: key });
      onUpdate(updatedUser);
      
      // Perform immediate health check
      const healthRes = type === 'groq_api_key' 
        ? await api.checkGroqHealth(key)
        : await api.checkJinaHealth(key);
      
      if (!healthRes.isActive) {
        setError(healthRes.error || 'Validation failed');
      } else {
        setShowCheck(true);
        setTimeout(() => {
          setShowCheck(false);
          setIsOpen(false);
        }, 1500);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to update key');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-zinc-100 dark:bg-[#27272a] border border-zinc-200 dark:border-white/10 hover:bg-zinc-200 dark:hover:bg-zinc-700 transition-colors group"
      >
        <div className={cn("w-2 h-2 rounded-full animate-pulse", status === 'online' ? "bg-emerald-500" : "bg-red-500")} />
        <span className="text-xs font-medium text-zinc-600 dark:text-zinc-300">
          <span className="text-zinc-400 dark:text-zinc-500 mr-1">{label}:</span>
          <span className={cn(status === 'online' ? "text-emerald-600 dark:text-emerald-400" : "text-red-600 dark:text-red-400")}>
            {status.charAt(0).toUpperCase() + status.slice(1)}
          </span>
        </span>
        <ChevronDown className={cn("w-3 h-3 text-zinc-400 transition-transform duration-300", isOpen && "rotate-180")} />
      </button>

      {isOpen && (
        <div className="absolute top-full mt-2 right-0 w-72 bg-white dark:bg-[#18181b] rounded-2xl border border-zinc-200 dark:border-white/10 shadow-2xl z-50 p-4 animate-in fade-in zoom-in-95 duration-200">
          <div className="flex items-center gap-2 mb-3">
            <Key className="w-4 h-4 text-violet-500" />
            <h3 className="text-sm font-bold text-zinc-900 dark:text-white">{label} Configuration</h3>
          </div>
          
          <div className="space-y-3">
            <div className="relative">
              <input 
                type={showKey ? "text" : "password"}
                value={key}
                onChange={(e) => setKey(e.target.value)}
                placeholder={`${label} API Key...`}
                className="w-full bg-zinc-50 dark:bg-white/5 border border-zinc-200 dark:border-white/10 rounded-xl py-2 pl-3 pr-10 text-xs text-zinc-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50 font-mono"
              />
              <button 
                onClick={() => setShowKey(!showKey)}
                className="absolute right-3 top-2 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200"
              >
                {showKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
              </button>
            </div>

            {error && (
              <div className="flex items-center gap-2 p-2 bg-red-500/10 rounded-lg text-[10px] text-red-500">
                <AlertCircle className="w-3 h-3 flex-shrink-0" />
                <span className="truncate">{error}</span>
              </div>
            )}

            <button 
              onClick={handleSave}
              disabled={isSaving}
              className={cn(
                "w-full flex items-center justify-center gap-2 py-2 rounded-xl text-xs font-bold transition-all active:scale-[0.98] disabled:opacity-70",
                showCheck ? "bg-emerald-500 text-white" : "bg-violet-600 hover:bg-violet-700 text-white shadow-lg shadow-violet-500/20"
              )}
            >
              {isSaving ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : showCheck ? (
                <Check className="w-3.5 h-3.5" />
              ) : (
                <Save className="w-3.5 h-3.5" />
              )}
              {isSaving ? 'Validating...' : showCheck ? 'Verified!' : 'Save & Verify'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

type View = 'dashboard' | 'scraper' | 'qagen' | 'config' | 'datasets' | 'settings' | 'signup' | 'login';

export default function App() {
  const [theme, setTheme] = useState<Theme>('dark');
  const [isLoading, setIsLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<any>(null);
  const [currentView, setCurrentView] = useState<View>('login');
  
  // Notifications State
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [isNotificationDrawerOpen, setIsNotificationDrawerOpen] = useState(false);
  
  // State for API data
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [qualityData, setQualityData] = useState<QualityDataPoint[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);

  // Theme Management
  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove('dark', 'theme-obsidian');
    
    if (theme === 'dark') {
      root.classList.add('dark');
    }
  }, [theme]);

  // Check for existing token and fetch user
  useEffect(() => {
    const fetchUser = async () => {
      const token = localStorage.getItem('token');
      if (token) {
        try {
          const userData = await api.getMe();
          setUser(userData);
          
          // Initialize status as offline until verified by health check
          setStatus({
            services: {
              jina: 'offline',
              groq: 'offline'
            },
            pipeline: {
              status: 'idle',
              currentStepId: null,
              steps: []
            }
          });

          setIsAuthenticated(true);
          setCurrentView('dashboard');
        } catch (err) {
          console.error("Session expired or invalid token", err);
          localStorage.removeItem('token');
          setIsAuthenticated(false);
          setCurrentView('login');
        }
      }
      setIsLoading(false);
    };
    
    fetchUser();
  }, []);

  // Auto Health Check Logic
  const runAutoHealthChecks = async (userData: any) => {
    if (!userData) return;

    // Check Groq if key exists
    if (userData.groq_api_key) {
      api.checkGroqHealth().then(res => {
        setStatus(prev => prev ? {
          ...prev,
          services: { ...prev.services, groq: res.isActive ? 'online' : 'offline' }
        } : prev);
      }).catch(() => {
        setStatus(prev => prev ? {
          ...prev,
          services: { ...prev.services, groq: 'offline' }
        } : prev);
      });
    }

    // Check Jina if key exists
    if (userData.jina_api_key) {
      api.checkJinaHealth().then(res => {
        setStatus(prev => prev ? {
          ...prev,
          services: { ...prev.services, jina: res.isActive ? 'online' : 'offline' }
        } : prev);
      }).catch(() => {
        setStatus(prev => prev ? {
          ...prev,
          services: { ...prev.services, jina: 'offline' }
        } : prev);
      });
    }
  };

  // Trigger health checks on login or key change
  useEffect(() => {
    if (isAuthenticated && user) {
      runAutoHealthChecks(user);
    }
  }, [isAuthenticated, user?.jina_api_key, user?.groq_api_key]);

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
        }
      };

      fetchData();
    }
  }, [isAuthenticated]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    setIsAuthenticated(false);
    setUser(null);
    setCurrentView('login');
  };

  const addNotification = (type: Notification['type'], title: string, message: string) => {
    const newNotif: Notification = {
      id: Math.random().toString(36).substr(2, 9),
      type,
      title,
      message,
      timestamp: new Date(),
      read: false
    };
    setNotifications(prev => [newNotif, ...prev]);
  };

  const markNotificationRead = (id: string) => {
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
  };

  const clearAllNotifications = () => {
    setNotifications([]);
  };

  const handleRunPipeline = async () => {
    try {
      addNotification('info', 'Pipeline Started', 'Initializing the data pipeline configuration...');
      await api.runPipeline({ configId: 'default-v1' });
      // In a real app, you'd probably poll for status updates here
      const newStatus = await api.getStatus();
      setStatus(newStatus);
      addNotification('success', 'Pipeline Active', 'The scraping phase has begun successfully.');
    } catch (error) {
      addNotification('error', 'Pipeline Failed', 'Could not establish connection with the background worker.');
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
        onLoginSuccess={(data: any) => {
          setIsAuthenticated(true);
          setUser(data.user);
          
          // Set initial status based on user keys
          setStatus({
            services: {
              jina: data.user.jina_api_key ? 'online' : 'offline',
              groq: data.user.groq_api_key ? 'online' : 'offline'
            },
            pipeline: {
              status: 'idle',
              currentStepId: null,
              steps: []
            }
          });

          setCurrentView('dashboard');
        }}
      />
    );
  }

  // Loading Screen for Dashboard Data
  if (!status) {
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
            user={user}
            onUserUpdate={(updatedUser: any) => setUser(updatedUser)}
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
    <div className="flex h-screen w-full overflow-hidden bg-[var(--bg-main)] text-[var(--text-main)]">
      {/* Sidebar - 30% Logical Weight */}
      <aside className="w-20 flex-shrink-0 border-r border-zinc-200 dark:border-white/10 bg-[var(--bg-surface)] flex flex-col items-center py-6 z-20">
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
          <ThemeSelector 
            currentTheme={theme} 
            onThemeChange={(newTheme) => setTheme(newTheme)} 
          />
          <button 
            onClick={handleLogout}
            className="w-8 h-8 rounded-full bg-[var(--brand-accent)] flex items-center justify-center text-white font-bold text-xs ring-2 ring-white dark:ring-[#18181b] hover:scale-110 transition-transform"
            title="Log Out"
          >
            {user?.first_name?.charAt(0) || user?.email?.charAt(0) || 'U'}
          </button>
        </div>
      </aside>

      {/* Main Content - 60% Logical Weight */}
      <main className="flex-1 flex flex-col h-full overflow-hidden relative bg-[var(--bg-main)]">
        {/* Header - Glassmorphism UI */}
        <header className="h-20 flex-shrink-0 px-8 flex items-center justify-between border-b border-zinc-200 dark:border-white/10 bg-[var(--bg-surface)]/80 backdrop-blur-md z-10">
          <div>
            <h1 className="text-xl font-bold text-[var(--text-main)]">QA Data Pipeline</h1>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 font-medium tracking-tight">AI Orchestration Mainframe</p>
          </div>

          <div className="flex items-center gap-4 relative">
            <StatusBadgeDropdown 
              label="Jina" 
              status={status?.services.jina || 'offline'} 
              user={user}
              type="jina_api_key"
              onUpdate={(updatedUser) => setUser(updatedUser)}
            />
            <StatusBadgeDropdown 
              label="Groq" 
              status={status?.services.groq || 'offline'} 
              user={user}
              type="groq_api_key"
              onUpdate={(updatedUser) => setUser(updatedUser)}
            />
            
            <div className="relative">
              <button 
                onClick={() => setIsNotificationDrawerOpen(!isNotificationDrawerOpen)}
                className={cn(
                  "p-2 text-zinc-500 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-white/5 rounded-xl transition-colors relative",
                  isNotificationDrawerOpen && "bg-zinc-100 dark:bg-white/5 text-[var(--brand-accent)]"
                )}
              >
                <Bell className="w-5 h-5" />
                {notifications.some(n => !n.read) && (
                  <span className="absolute top-2 right-2 w-2 h-2 bg-red-500 rounded-full border-2 border-white dark:border-[#09090b]" />
                )}
              </button>

              {isNotificationDrawerOpen && (
                <NotificationDrawer 
                  notifications={notifications}
                  onMarkRead={markNotificationRead}
                  onClearAll={clearAllNotifications}
                  onClose={() => setIsNotificationDrawerOpen(false)}
                />
              )}
            </div>

            <button 
              onClick={handleRunPipeline}
              className="flex items-center gap-2 bg-[var(--brand-accent)] hover:opacity-90 text-white px-5 py-2.5 rounded-xl font-bold shadow-lg shadow-violet-500/20 transition-all active:scale-95"
            >
              <Play className="w-4 h-4 fill-current" />
              RUN PIPELINE
            </button>
          </div>
        </header>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-8 scrollbar-hide">
          <div className="max-w-[1400px] mx-auto space-y-6">
            {renderContent()}
          </div>
        </div>
      </main>
    </div>
  );
}
