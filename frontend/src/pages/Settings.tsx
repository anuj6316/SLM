import React, { useState } from 'react';
import { User, Mail, Shield, LogOut, Monitor, Bell, Globe, Check, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

export default function Settings() {
  const [activeTab, setActiveTab] = useState('profile');
  const [twoFactorEnabled, setTwoFactorEnabled] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [showSaved, setShowSaved] = useState(false);

  const handleSave = () => {
    setIsSaving(true);
    setTimeout(() => {
      setIsSaving(false);
      setShowSaved(true);
      setTimeout(() => setShowSaved(false), 2000);
    }, 1000);
  };

  return (
    <div className="space-y-8 max-w-[1000px] mx-auto">
        {/* Header */}
        <div>
            <h1 className="text-3xl font-bold text-zinc-900 dark:text-white">Settings</h1>
            <p className="text-zinc-500 dark:text-zinc-400 mt-1">Manage your account and preferences</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Sidebar Navigation */}
            <div className="lg:col-span-1">
                <div className="bg-white dark:bg-[#18181b] rounded-3xl border border-zinc-100 dark:border-white/5 p-4 shadow-sm">
                    <nav className="space-y-1">
                        <button 
                            onClick={() => setActiveTab('profile')}
                            className={cn(
                                "w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-colors",
                                activeTab === 'profile' 
                                    ? "bg-violet-50 dark:bg-violet-500/10 text-violet-600 dark:text-violet-400" 
                                    : "text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-white/5"
                            )}
                        >
                            <User className="w-4 h-4" />
                            Profile
                        </button>
                        <button 
                            onClick={() => setActiveTab('appearance')}
                            className={cn(
                                "w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-colors",
                                activeTab === 'appearance' 
                                    ? "bg-violet-50 dark:bg-violet-500/10 text-violet-600 dark:text-violet-400" 
                                    : "text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-white/5"
                            )}
                        >
                            <Monitor className="w-4 h-4" />
                            Appearance
                        </button>
                        <button 
                            onClick={() => setActiveTab('notifications')}
                            className={cn(
                                "w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-colors",
                                activeTab === 'notifications' 
                                    ? "bg-violet-50 dark:bg-violet-500/10 text-violet-600 dark:text-violet-400" 
                                    : "text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-white/5"
                            )}
                        >
                            <Bell className="w-4 h-4" />
                            Notifications
                        </button>
                        <button 
                            onClick={() => setActiveTab('language')}
                            className={cn(
                                "w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-colors",
                                activeTab === 'language' 
                                    ? "bg-violet-50 dark:bg-violet-500/10 text-violet-600 dark:text-violet-400" 
                                    : "text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-white/5"
                            )}
                        >
                            <Globe className="w-4 h-4" />
                            Language
                        </button>
                        <div className="h-px bg-zinc-100 dark:bg-white/5 my-2" />
                        <button className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-red-600 hover:bg-red-50 dark:hover:bg-red-500/10 transition-colors">
                            <LogOut className="w-4 h-4" />
                            Sign Out
                        </button>
                    </nav>
                </div>
            </div>

            {/* Main Content */}
            <div className="lg:col-span-2 space-y-6">
                
                {/* Profile Card */}
                <div className={cn("bg-white dark:bg-[#18181b] rounded-3xl border border-zinc-100 dark:border-white/5 p-8 shadow-sm", activeTab !== 'profile' && 'hidden')}>
                    <h2 className="text-lg font-bold text-zinc-900 dark:text-white mb-6">Profile Information</h2>
                    
                    <div className="flex items-center gap-6 mb-8">
                        <div className="w-20 h-20 rounded-full bg-violet-500 flex items-center justify-center text-white text-2xl font-bold border-4 border-white dark:border-[#18181b] shadow-lg">
                            JD
                        </div>
                        <div>
                            <button className="px-4 py-2 rounded-xl bg-zinc-900 dark:bg-white text-white dark:text-zinc-900 text-sm font-bold hover:opacity-90 transition-opacity">
                                Change Avatar
                            </button>
                            <p className="text-xs text-zinc-500 mt-2">JPG, GIF or PNG. Max size 800K</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-2">
                            <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">First Name</label>
                            <input 
                                type="text" 
                                defaultValue="John" 
                                className="w-full bg-zinc-50 dark:bg-white/5 border border-zinc-200 dark:border-white/10 rounded-xl py-3 px-4 text-sm text-zinc-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                            />
                        </div>
                        <div className="space-y-2">
                            <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Last Name</label>
                            <input 
                                type="text" 
                                defaultValue="Doe" 
                                className="w-full bg-zinc-50 dark:bg-white/5 border border-zinc-200 dark:border-white/10 rounded-xl py-3 px-4 text-sm text-zinc-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                            />
                        </div>
                        <div className="space-y-2 md:col-span-2">
                            <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Email Address</label>
                            <div className="relative">
                                <Mail className="absolute left-4 top-3.5 w-4 h-4 text-zinc-400" />
                                <input 
                                    type="email" 
                                    defaultValue="john.doe@example.com" 
                                    className="w-full bg-zinc-50 dark:bg-white/5 border border-zinc-200 dark:border-white/10 rounded-xl py-3 pl-10 pr-4 text-sm text-zinc-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                />
                            </div>
                        </div>
                    </div>

                    <div className="mt-8 flex justify-end">
                        <button 
                            onClick={handleSave}
                            disabled={isSaving || showSaved}
                            className={cn(
                                "px-6 py-3 rounded-xl font-bold transition-colors shadow-lg active:scale-95 disabled:opacity-70 disabled:cursor-not-allowed flex items-center gap-2",
                                showSaved 
                                    ? "bg-emerald-500 hover:bg-emerald-600 text-white shadow-emerald-500/20" 
                                    : "bg-violet-600 hover:bg-violet-700 text-white shadow-violet-500/20"
                            )}
                        >
                            {isSaving ? (
                                <Loader2 className="w-4 h-4 animate-spin" />
                            ) : showSaved ? (
                                <Check className="w-4 h-4" />
                            ) : null}
                            {isSaving ? 'Saving...' : showSaved ? 'Saved!' : 'Save Changes'}
                        </button>
                    </div>
                </div>

                {/* Security Card */}
                <div className={cn("bg-white dark:bg-[#18181b] rounded-3xl border border-zinc-100 dark:border-white/5 p-8 shadow-sm", activeTab !== 'profile' && 'hidden')}>
                    <div className="flex items-center gap-3 mb-6">
                        <Shield className="w-5 h-5 text-zinc-400" />
                        <h2 className="text-lg font-bold text-zinc-900 dark:text-white">Security</h2>
                    </div>
                    
                    <div className="flex items-center justify-between py-4 border-b border-zinc-100 dark:border-white/5">
                        <div>
                            <div className="font-medium text-zinc-900 dark:text-white">Two-Factor Authentication</div>
                            <div className="text-xs text-zinc-500">Add an extra layer of security to your account</div>
                        </div>
                        <button 
                            onClick={() => setTwoFactorEnabled(!twoFactorEnabled)}
                            className={cn(
                                "px-4 py-2 rounded-lg border text-sm font-medium transition-colors",
                                twoFactorEnabled 
                                    ? "bg-emerald-500 border-emerald-500 text-white hover:bg-emerald-600" 
                                    : "border-zinc-200 dark:border-white/10 hover:bg-zinc-50 dark:hover:bg-white/5"
                            )}
                        >
                            {twoFactorEnabled ? 'Enabled' : 'Enable'}
                        </button>
                    </div>
                    
                    <div className="flex items-center justify-between py-4">
                        <div>
                            <div className="font-medium text-zinc-900 dark:text-white">Password</div>
                            <div className="text-xs text-zinc-500">Last changed 3 months ago</div>
                        </div>
                        <button className="px-4 py-2 rounded-lg border border-zinc-200 dark:border-white/10 text-sm font-medium hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors">
                            Change
                        </button>
                    </div>
                </div>

                {/* Placeholder for other tabs */}
                <div className={cn("bg-white dark:bg-[#18181b] rounded-3xl border border-zinc-100 dark:border-white/5 p-12 shadow-sm flex flex-col items-center justify-center text-center", (activeTab === 'profile' || activeTab === 'settings') && 'hidden')}>
                    <div className="w-16 h-16 rounded-full bg-zinc-100 dark:bg-white/5 flex items-center justify-center mb-4">
                        <Monitor className="w-8 h-8 text-zinc-400" />
                    </div>
                    <h3 className="text-lg font-bold text-zinc-900 dark:text-white">Coming Soon</h3>
                    <p className="text-zinc-500 dark:text-zinc-400 mt-2 max-w-xs">This section is currently under development and will be available in a future update.</p>
                </div>

            </div>
        </div>
    </div>
  );
}
