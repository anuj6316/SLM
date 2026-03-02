import React, { useState } from 'react';
import { Play, Plus, Globe, Clock, CheckCircle2, AlertCircle, Loader2, Filter, Layers } from 'lucide-react';
import { cn } from '@/lib/utils';

// Mock data
const recentScrapes = [
  { id: 1, url: 'https://example.com/docs', status: 'completed', pages: 142, duration: '4m 20s', date: '2 mins ago' },
  { id: 2, url: 'https://api.service.io', status: 'active', pages: 45, duration: '1m 12s', date: 'Now' },
  { id: 3, url: 'https://blog.tech.net', status: 'failed', pages: 12, duration: '45s', date: '1 hour ago' },
  { id: 4, url: 'https://docs.framework.dev', status: 'completed', pages: 890, duration: '12m 05s', date: '3 hours ago' },
  { id: 5, url: 'https://company.site/about', status: 'completed', pages: 5, duration: '10s', date: '5 hours ago' },
];

export default function Scraper() {
  const [url, setUrl] = useState('');
  const [maxDepth, setMaxDepth] = useState(2);
  const [concurrency, setConcurrency] = useState(5);
  const [scrapeType, setScrapeType] = useState<'flash' | 'deep'>('flash');

  return (
    <div className="space-y-8 max-w-[1600px] mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center">
            <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Web Scraper</h1>
                <p className="text-gray-500 dark:text-gray-400 mt-1">Manage extraction tasks and sources</p>
            </div>
            <button className="flex items-center gap-2 bg-violet-600 hover:bg-violet-700 text-white px-5 py-2.5 rounded-xl font-medium transition-all shadow-lg shadow-violet-500/20 active:scale-95">
                <Plus className="w-4 h-4" />
                New Job
            </button>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
            {/* Configuration Card */}
            <div className="xl:col-span-1 space-y-6">
                <div className="bg-white dark:bg-[#15171C] rounded-3xl border border-gray-100 dark:border-white/5 p-8 shadow-sm">
                    <h2 className="text-lg font-bold text-gray-900 dark:text-white mb-6">Quick Scrape</h2>
                    <div className="space-y-5">
                        <div className="space-y-2">
                            <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Target URL</label>
                            <div className="relative">
                                <Globe className="absolute left-4 top-3.5 w-4 h-4 text-gray-400" />
                                <input 
                                    type="text" 
                                    value={url}
                                    onChange={(e) => setUrl(e.target.value)}
                                    placeholder="https://example.com" 
                                    className="w-full bg-gray-50 dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-xl py-3 pl-10 pr-4 text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                />
                            </div>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Max Depth</label>
                                <input 
                                    type="number" 
                                    value={maxDepth}
                                    onChange={(e) => setMaxDepth(parseInt(e.target.value))}
                                    className="w-full bg-gray-50 dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-xl py-3 px-4 text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                />
                            </div>
                            <div className="space-y-2">
                                <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Concurrency</label>
                                <input 
                                    type="number" 
                                    value={concurrency}
                                    onChange={(e) => setConcurrency(parseInt(e.target.value))}
                                    className="w-full bg-gray-50 dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-xl py-3 px-4 text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50"
                                />
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Scrape Type</label>
                            <div className="relative">
                                <Layers className="absolute left-4 top-3.5 w-4 h-4 text-gray-400" />
                                <select 
                                    value={scrapeType}
                                    onChange={(e) => setScrapeType(e.target.value as 'flash' | 'deep')}
                                    className="w-full bg-gray-50 dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-xl py-3 pl-10 pr-4 text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50 appearance-none cursor-pointer"
                                >
                                    <option value="flash">Flash Scrape (Fast, Single Page)</option>
                                    <option value="deep">Deep Scrape (Recursive, Multi-page)</option>
                                </select>
                            </div>
                        </div>

                        <div className="pt-2">
                            <button className="w-full flex items-center justify-center gap-2 bg-gray-900 dark:bg-white text-white dark:text-gray-900 py-3 rounded-xl font-bold hover:opacity-90 transition-opacity">
                                <Play className="w-4 h-4 fill-current" />
                                Start Extraction
                            </button>
                        </div>
                    </div>
                </div>

                {/* Stats */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="bg-violet-500 rounded-3xl p-6 text-white shadow-lg shadow-violet-500/20">
                        <div className="text-violet-200 text-xs font-medium uppercase tracking-wider mb-1">Pages Scraped</div>
                        <div className="text-3xl font-bold">12.5k</div>
                    </div>
                    <div className="bg-white dark:bg-[#15171C] rounded-3xl border border-gray-100 dark:border-white/5 p-6">
                        <div className="text-gray-500 text-xs font-medium uppercase tracking-wider mb-1">Active Jobs</div>
                        <div className="text-3xl font-bold text-gray-900 dark:text-white">3</div>
                    </div>
                </div>
            </div>

            {/* Recent Jobs List */}
            <div className="xl:col-span-2 bg-white dark:bg-[#15171C] rounded-3xl border border-gray-100 dark:border-white/5 p-8 shadow-sm flex flex-col">
                <div className="flex items-center justify-between mb-8">
                    <h2 className="text-lg font-bold text-gray-900 dark:text-white">Recent Jobs</h2>
                    <div className="flex gap-2">
                        <button className="p-2 text-gray-400 hover:text-violet-500 hover:bg-violet-50 dark:hover:bg-white/5 rounded-lg transition-colors">
                            <Filter className="w-4 h-4" />
                        </button>
                    </div>
                </div>
                
                <div className="space-y-4 flex-1 overflow-y-auto pr-2">
                    {recentScrapes.map((job) => (
                        <div key={job.id} className="group flex items-center justify-between p-4 rounded-2xl border border-gray-100 dark:border-white/5 hover:border-violet-500/30 hover:bg-gray-50 dark:hover:bg-white/5 transition-all cursor-pointer">
                            <div className="flex items-center gap-4">
                                <div className={cn(
                                    "w-12 h-12 rounded-2xl flex items-center justify-center transition-colors",
                                    job.status === 'completed' ? "bg-emerald-100 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400" :
                                    job.status === 'active' ? "bg-violet-100 text-violet-600 dark:bg-violet-500/10 dark:text-violet-400" :
                                    "bg-red-100 text-red-600 dark:bg-red-500/10 dark:text-red-400"
                                )}>
                                    {job.status === 'completed' ? <CheckCircle2 className="w-6 h-6" /> :
                                     job.status === 'active' ? <Loader2 className="w-6 h-6 animate-spin" /> :
                                     <AlertCircle className="w-6 h-6" />}
                                </div>
                                <div>
                                    <h3 className="font-semibold text-gray-900 dark:text-white text-base">{job.url}</h3>
                                    <div className="flex items-center gap-4 text-xs text-gray-500 mt-1">
                                        <span className="flex items-center gap-1.5"><Globe className="w-3.5 h-3.5" /> {job.pages} pages</span>
                                        <span className="flex items-center gap-1.5"><Clock className="w-3.5 h-3.5" /> {job.duration}</span>
                                    </div>
                                </div>
                            </div>
                            <div className="text-right">
                                <div className={cn(
                                    "text-xs font-bold uppercase tracking-wider mb-1",
                                    job.status === 'completed' ? "text-emerald-500" :
                                    job.status === 'active' ? "text-violet-500" :
                                    "text-red-500"
                                )}>{job.status}</div>
                                <div className="text-xs text-gray-400">{job.date}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    </div>
  );
}
