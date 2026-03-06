import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Play, Plus, Globe, Clock, CheckCircle2, AlertCircle, Loader2, Filter, Layers, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { api } from '@/services/api';
import { ScrapeJob } from '@/types';

export default function Scraper() {
    // --- Form State ---
    const [url, setUrl] = useState('');
    const [maxDepth, setMaxDepth] = useState(2);
    const [concurrency, setConcurrency] = useState(5);
    const [scrapeType, setScrapeType] = useState<'flash' | 'deep'>('flash');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [errorStatus, setErrorStatus] = useState<string | null>(null);
    const [successMsg, setSuccessMsg] = useState<string | null>(null);

    // --- Jobs State (REAL data from backend) ---
    const [jobs, setJobs] = useState<ScrapeJob[]>([]);
    const [isLoadingJobs, setIsLoadingJobs] = useState(true);
    const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

    // --- Fetch jobs from backend ---
    const fetchJobs = useCallback(async () => {
        try {
            const response = await api.getScrapeJobs();
            setJobs(response.jobs);
        } catch (err) {
            // Don't show error on poll failures — silent fail is better UX
            console.error('Failed to fetch jobs:', err);
        } finally {
            setIsLoadingJobs(false);
        }
    }, []);

    // --- Initial load + Smart Polling ---
    // Poll every 3s ONLY when there are active/pending jobs
    // Stop polling when all jobs are completed/failed (saves bandwidth)
    useEffect(() => {
        fetchJobs(); // Initial fetch

        return () => {
            if (pollingRef.current) clearInterval(pollingRef.current);
        };
    }, [fetchJobs]);

    useEffect(() => {
        const hasActiveJobs = jobs.some(j => j.status === 'active' || j.status === 'pending');

        if (hasActiveJobs && !pollingRef.current) {
            // Start polling
            pollingRef.current = setInterval(fetchJobs, 3000);
        } else if (!hasActiveJobs && pollingRef.current) {
            // Stop polling — nothing to watch
            clearInterval(pollingRef.current);
            pollingRef.current = null;
        }

        return () => {
            if (pollingRef.current) {
                clearInterval(pollingRef.current);
                pollingRef.current = null;
            }
        };
    }, [jobs, fetchJobs]);

    // --- Computed Stats (derived from real data) ---
    const totalPagesScraped = jobs.reduce((sum, j) => sum + j.pages_scraped, 0);
    const activeJobCount = jobs.filter(j => j.status === 'active' || j.status === 'pending').length;

    // --- Submit Handler ---
    const handleStartExtraction = async () => {
        if (!url) {
            setErrorStatus('URL is required');
            return;
        }

        try {
            setIsSubmitting(true);
            setErrorStatus(null);
            setSuccessMsg(null);

            const result = await api.runScrape({ url, maxDepth, concurrency, scrapeType });
            setSuccessMsg(`Job started! ID: ${result.job_id}`);
            setUrl(''); // Clear the input

            // Immediately refresh the job list so user sees new pending job
            await fetchJobs();
        } catch (error: any) {
            setErrorStatus(error.message || 'Failed to start scraping');
        } finally {
            setIsSubmitting(false);
        }
    };

    // --- Time Ago Helper ---
    const timeAgo = (dateStr: string) => {
        const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000);
        if (seconds < 60) return 'Just now';
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60) return `${minutes}m ago`;
        const hours = Math.floor(minutes / 60);
        if (hours < 24) return `${hours}h ago`;
        const days = Math.floor(hours / 24);
        return `${days}d ago`;
    };

    // --- Status config for icons/colors ---
    const statusConfig = {
        pending: { icon: Clock, color: 'bg-amber-100 text-amber-600 dark:bg-amber-500/10 dark:text-amber-400', textColor: 'text-amber-500' },
        active: { icon: Loader2, color: 'bg-violet-100 text-violet-600 dark:bg-violet-500/10 dark:text-violet-400', textColor: 'text-violet-500', animate: true },
        completed: { icon: CheckCircle2, color: 'bg-emerald-100 text-emerald-600 dark:bg-emerald-500/10 dark:text-emerald-400', textColor: 'text-emerald-500' },
        failed: { icon: AlertCircle, color: 'bg-red-100 text-red-600 dark:bg-red-500/10 dark:text-red-400', textColor: 'text-red-500' },
    };

    return (
        <div className="space-y-8 max-w-[1600px] mx-auto">
            {/* Header */}
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Web Scraper</h1>
                    <p className="text-gray-500 dark:text-gray-400 mt-1">Manage extraction tasks and sources</p>
                </div>
                <button
                    onClick={fetchJobs}
                    className="flex items-center gap-2 bg-violet-600 hover:bg-violet-700 text-white px-5 py-2.5 rounded-xl font-medium transition-all shadow-lg shadow-violet-500/20 active:scale-95"
                >
                    <RefreshCw className="w-4 h-4" />
                    Refresh
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
                                        onKeyDown={(e) => e.key === 'Enter' && handleStartExtraction()}
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
                                {errorStatus && (
                                    <div className="mb-4 text-sm text-red-500 flex items-center gap-2 bg-red-500/10 p-3 rounded-xl">
                                        <AlertCircle className="w-4 h-4 flex-shrink-0" />
                                        {errorStatus}
                                    </div>
                                )}
                                {successMsg && (
                                    <div className="mb-4 text-sm text-emerald-500 flex items-center gap-2 bg-emerald-500/10 p-3 rounded-xl">
                                        <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
                                        {successMsg}
                                    </div>
                                )}
                                <button
                                    onClick={handleStartExtraction}
                                    disabled={isSubmitting}
                                    className="w-full flex items-center justify-center gap-2 bg-gray-900 dark:bg-white text-white dark:text-gray-900 py-3 rounded-xl font-bold hover:opacity-90 transition-opacity disabled:opacity-50"
                                >
                                    {isSubmitting ? (
                                        <>
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                            Starting...
                                        </>
                                    ) : (
                                        <>
                                            <Play className="w-4 h-4 fill-current" />
                                            Start Extraction
                                        </>
                                    )}
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Stats — now using REAL computed data */}
                    <div className="grid grid-cols-2 gap-4">
                        <div className="bg-violet-500 rounded-3xl p-6 text-white shadow-lg shadow-violet-500/20">
                            <div className="text-violet-200 text-xs font-medium uppercase tracking-wider mb-1">Pages Scraped</div>
                            <div className="text-3xl font-bold">{totalPagesScraped.toLocaleString()}</div>
                        </div>
                        <div className="bg-white dark:bg-[#15171C] rounded-3xl border border-gray-100 dark:border-white/5 p-6 relative overflow-hidden">
                            <div className="text-gray-500 text-xs font-medium uppercase tracking-wider mb-1">Active Jobs</div>
                            <div className="text-3xl font-bold text-gray-900 dark:text-white">{activeJobCount}</div>
                            {activeJobCount > 0 && (
                                <div className="absolute top-3 right-3">
                                    <span className="relative flex h-3 w-3">
                                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-violet-400 opacity-75"></span>
                                        <span className="relative inline-flex rounded-full h-3 w-3 bg-violet-500"></span>
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Recent Jobs List — now using REAL data */}
                <div className="xl:col-span-2 bg-white dark:bg-[#15171C] rounded-3xl border border-gray-100 dark:border-white/5 p-8 shadow-sm flex flex-col">
                    <div className="flex items-center justify-between mb-8">
                        <div className="flex items-center gap-3">
                            <h2 className="text-lg font-bold text-gray-900 dark:text-white">Recent Jobs</h2>
                            {activeJobCount > 0 && (
                                <span className="text-[10px] font-bold text-violet-500 bg-violet-500/10 px-2 py-0.5 rounded-full uppercase tracking-wider animate-pulse">
                                    {activeJobCount} running
                                </span>
                            )}
                        </div>
                        <div className="flex gap-2">
                            <button
                                onClick={fetchJobs}
                                className="p-2 text-gray-400 hover:text-violet-500 hover:bg-violet-50 dark:hover:bg-white/5 rounded-lg transition-colors"
                                title="Refresh jobs"
                            >
                                <RefreshCw className="w-4 h-4" />
                            </button>
                        </div>
                    </div>

                    <div className="space-y-4 flex-1 overflow-y-auto pr-2">
                        {isLoadingJobs ? (
                            <div className="flex flex-col items-center justify-center py-16 gap-3">
                                <Loader2 className="w-8 h-8 animate-spin text-violet-500" />
                                <p className="text-sm text-gray-400">Loading jobs...</p>
                            </div>
                        ) : jobs.length === 0 ? (
                            <div className="flex flex-col items-center justify-center py-16 gap-3">
                                <div className="w-16 h-16 rounded-2xl bg-gray-100 dark:bg-white/5 flex items-center justify-center">
                                    <Globe className="w-8 h-8 text-gray-300 dark:text-gray-600" />
                                </div>
                                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">No scrape jobs yet</p>
                                <p className="text-xs text-gray-400">Use the form to start your first extraction!</p>
                            </div>
                        ) : (
                            jobs.map((job) => {
                                const config = statusConfig[job.status];
                                const StatusIcon = config.icon;

                                return (
                                    <div key={job.job_id} className="group flex items-center justify-between p-4 rounded-2xl border border-gray-100 dark:border-white/5 hover:border-violet-500/30 hover:bg-gray-50 dark:hover:bg-white/5 transition-all cursor-pointer">
                                        <div className="flex items-center gap-4">
                                            <div className={cn("w-12 h-12 rounded-2xl flex items-center justify-center transition-colors", config.color)}>
                                                <StatusIcon className={cn("w-6 h-6", config.animate && "animate-spin")} />
                                            </div>
                                            <div>
                                                <h3 className="font-semibold text-gray-900 dark:text-white text-base truncate max-w-[350px]">{job.url}</h3>
                                                <div className="flex items-center gap-4 text-xs text-gray-500 mt-1">
                                                    <span className="flex items-center gap-1.5">
                                                        <Globe className="w-3.5 h-3.5" /> {job.pages_scraped} pages
                                                    </span>
                                                    <span className="flex items-center gap-1.5">
                                                        <Layers className="w-3.5 h-3.5" /> {job.scrape_type}
                                                    </span>
                                                </div>
                                                {job.error_message && (
                                                    <p className="text-xs text-red-400 mt-1 truncate max-w-[350px]">{job.error_message}</p>
                                                )}
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className={cn("text-xs font-bold uppercase tracking-wider mb-1", config.textColor)}>
                                                {job.status}
                                            </div>
                                            <div className="text-xs text-gray-400">{timeAgo(job.created_at)}</div>
                                        </div>
                                    </div>
                                );
                            })
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
