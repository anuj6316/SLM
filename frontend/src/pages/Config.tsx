import React, { useState } from 'react';
import { Key, Shield, Database, Bell, Save, Eye, EyeOff, Loader2, Check, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useHealth } from '@/hooks/useHealth';
import { SystemStatus } from '@/types';

export default function Config({ onStatusUpdate }: { onStatusUpdate: (status: SystemStatus) => void }) {
  const { checkStatus, isLoading: isCheckingHealth, error: healthError, data: healthData } = useHealth();
  
  const [showGroqKey, setShowGroqKey] = useState(false);
  const [showJinaKey, setShowJinaKey] = useState(false);
  
  const [groqKey, setGroqKey] = useState('gsk_89234789234...');
  const [jinaKey, setJinaKey] = useState('jina_v2_...');

  const [notifyCompletion, setNotifyCompletion] = useState(false);
  const [notifyError, setNotifyError] = useState(true);

  const [isSaving, setIsSaving] = useState(false);
  const [showSaved, setShowSaved] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const result = await checkStatus({
        groq_api_key: groqKey,
        jina_api_key: jinaKey
      });
      
      onStatusUpdate({
        services: {
          jina: result.jina_isActive ? 'online' : 'offline',
          groq: result.groq_isActive ? 'online' : 'offline'
        },
        pipeline: {
            status: 'idle',
            currentStepId: null,
            steps: []
        }
      });
      
      setShowSaved(true);
      setTimeout(() => setShowSaved(false), 2000);
    } catch (err) {
      console.error("Failed to validate keys", err);
    } finally {
      setIsSaving(false);
    }
  };

  const jinaConnected = healthData?.jina_isActive ?? false;
  const groqConnected = healthData?.groq_isActive ?? false;

  return (
    <div className="space-y-8 max-w-[1000px] mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center">
            <div>
                <h1 className="text-3xl font-bold text-zinc-900 dark:text-white">Configuration</h1>
                <p className="text-zinc-500 dark:text-zinc-400 mt-1">Manage API keys and system connections</p>
            </div>
            <button 
                onClick={handleSave}
                disabled={isSaving || showSaved}
                className={cn(
                    "flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium transition-all shadow-lg active:scale-95 disabled:opacity-70 disabled:cursor-not-allowed",
                    showSaved 
                        ? "bg-emerald-500 hover:bg-emerald-600 text-white shadow-emerald-500/20" 
                        : "bg-violet-600 hover:bg-violet-700 text-white shadow-violet-500/20"
                )}
            >
                {isSaving ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                ) : showSaved ? (
                    <Check className="w-4 h-4" />
                ) : (
                    <Save className="w-4 h-4" />
                )}
                {isSaving ? 'Saving...' : showSaved ? 'Saved!' : 'Save Changes'}
            </button>
        </div>

        <div className="space-y-6">
            
            {/* API Keys Section */}
            <div className="bg-white dark:bg-[#18181b] rounded-3xl border border-zinc-100 dark:border-white/5 p-8 shadow-sm">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-violet-100 dark:bg-violet-500/10 rounded-lg text-violet-600 dark:text-violet-400">
                        <Key className="w-5 h-5" />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-zinc-900 dark:text-white">API Keys</h2>
                        <p className="text-sm text-zinc-500 dark:text-zinc-400">Securely manage your provider credentials</p>
                    </div>
                </div>

                <div className="space-y-6">
                    {healthError && (
                        <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-center gap-3 text-red-500 text-sm">
                            <AlertCircle className="w-4 h-4" />
                            {healthError}
                        </div>
                    )}
                    <div className="space-y-2">
                        <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Groq API Key</label>
                        <div className="relative">
                            <input 
                                type={showGroqKey ? "text" : "password"}
                                value={groqKey}
                                onChange={(e) => setGroqKey(e.target.value)}
                                className="w-full bg-zinc-50 dark:bg-white/5 border border-zinc-200 dark:border-white/10 rounded-xl py-3 pl-4 pr-12 text-sm text-zinc-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50 font-mono"
                            />
                            <button 
                                onClick={() => setShowGroqKey(!showGroqKey)}
                                className="absolute right-4 top-3 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors"
                            >
                                {showGroqKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                            </button>
                        </div>
                    </div>

                    <div className="space-y-2">
                        <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Jina AI Key</label>
                        <div className="relative">
                            <input 
                                type={showJinaKey ? "text" : "password"}
                                value={jinaKey}
                                onChange={(e) => setJinaKey(e.target.value)}
                                className="w-full bg-zinc-50 dark:bg-white/5 border border-zinc-200 dark:border-white/10 rounded-xl py-3 pl-4 pr-12 text-sm text-zinc-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50 font-mono"
                            />
                            <button 
                                onClick={() => setShowJinaKey(!showJinaKey)}
                                className="absolute right-4 top-3 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors"
                            >
                                {showJinaKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Services Section */}
            <div className="bg-white dark:bg-[#18181b] rounded-3xl border border-zinc-100 dark:border-white/5 p-8 shadow-sm">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-emerald-100 dark:bg-emerald-500/10 rounded-lg text-emerald-600 dark:text-emerald-400">
                        <Database className="w-5 h-5" />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-zinc-900 dark:text-white">Service Connections</h2>
                        <p className="text-sm text-zinc-500 dark:text-zinc-400">Manage external integrations</p>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="p-4 rounded-2xl border border-zinc-200 dark:border-white/10 bg-zinc-50 dark:bg-white/5 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-white dark:bg-black/20 flex items-center justify-center text-lg font-bold">J</div>
                            <div>
                                <div className="font-bold text-zinc-900 dark:text-white">Jina Reader</div>
                                <div className={cn("text-xs font-medium", jinaConnected ? "text-emerald-500" : "text-zinc-500")}>
                                    {jinaConnected ? 'Connected' : 'Disconnected'}
                                </div>
                            </div>
                        </div>
                        <div className={cn(
                                "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
                                jinaConnected ? "bg-violet-600" : "bg-zinc-200 dark:bg-zinc-700"
                            )}>
                            <span className={cn(
                                "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                                jinaConnected ? "translate-x-6" : "translate-x-1"
                            )} />
                        </div>
                    </div>

                    <div className="p-4 rounded-2xl border border-zinc-200 dark:border-white/10 bg-zinc-50 dark:bg-white/5 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-white dark:bg-black/20 flex items-center justify-center text-lg font-bold">G</div>
                            <div>
                                <div className="font-bold text-zinc-900 dark:text-white">Groq Cloud</div>
                                <div className={cn("text-xs font-medium", groqConnected ? "text-emerald-500" : "text-zinc-500")}>
                                    {groqConnected ? 'Connected' : 'Disconnected'}
                                </div>
                            </div>
                        </div>
                        <div className={cn(
                                "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
                                groqConnected ? "bg-violet-600" : "bg-zinc-200 dark:bg-zinc-700"
                            )}>
                            <span className={cn(
                                "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                                groqConnected ? "translate-x-6" : "translate-x-1"
                            )} />
                        </div>
                    </div>
                </div>
            </div>

            {/* Notifications Section */}
            <div className="bg-white dark:bg-[#18181b] rounded-3xl border border-zinc-100 dark:border-white/5 p-8 shadow-sm">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-amber-100 dark:bg-amber-500/10 rounded-lg text-amber-600 dark:text-amber-400">
                        <Bell className="w-5 h-5" />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-zinc-900 dark:text-white">Notifications</h2>
                        <p className="text-sm text-zinc-500 dark:text-zinc-400">Alert preferences</p>
                    </div>
                </div>

                <div className="space-y-4">
                    <div className="flex items-center justify-between py-2">
                        <div>
                            <div className="font-medium text-zinc-900 dark:text-white">Pipeline Completion</div>
                            <div className="text-xs text-zinc-500">Notify when a job finishes successfully</div>
                        </div>
                        <button 
                            onClick={() => setNotifyCompletion(!notifyCompletion)}
                            className={cn(
                                "relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-violet-500 focus:ring-offset-2 dark:focus:ring-offset-zinc-900",
                                notifyCompletion ? "bg-violet-600" : "bg-zinc-200 dark:bg-zinc-700"
                            )}
                        >
                            <span className={cn(
                                "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                                notifyCompletion ? "translate-x-6" : "translate-x-1"
                            )} />
                        </button>
                    </div>
                    <div className="flex items-center justify-between py-2 border-t border-zinc-100 dark:border-white/5">
                        <div>
                            <div className="font-medium text-zinc-900 dark:text-white">Error Alerts</div>
                            <div className="text-xs text-zinc-500">Notify immediately on critical failures</div>
                        </div>
                        <button 
                            onClick={() => setNotifyError(!notifyError)}
                            className={cn(
                                "relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-violet-500 focus:ring-offset-2 dark:focus:ring-offset-zinc-900",
                                notifyError ? "bg-violet-600" : "bg-zinc-200 dark:bg-zinc-700"
                            )}
                        >
                            <span className={cn(
                                "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
                                notifyError ? "translate-x-6" : "translate-x-1"
                            )} />
                        </button>
                    </div>
                </div>
            </div>

        </div>
    </div>
  );
}
