import React from 'react';
import { Download, Trash2, Eye, FileJson, FileText, Search, Filter, MoreHorizontal } from 'lucide-react';
import { cn } from '@/lib/utils';

// Mock data
const datasets = [
  { id: 1, name: 'Unstructured to QA (Batch A)', type: 'JSON', size: '2.4 MB', rows: 450, created: '2 hours ago', status: 'Ready' },
  { id: 2, name: 'Documentation Scrape v2', type: 'CSV', size: '1.1 MB', rows: 210, created: '5 hours ago', status: 'Ready' },
  { id: 3, name: 'Blog Posts Extraction', type: 'JSON', size: '850 KB', rows: 120, created: '1 day ago', status: 'Archived' },
  { id: 4, name: 'Support Tickets Analysis', type: 'JSON', size: '5.6 MB', rows: 1200, created: '2 days ago', status: 'Ready' },
  { id: 5, name: 'Product Manuals QA', type: 'CSV', size: '3.2 MB', rows: 680, created: '3 days ago', status: 'Ready' },
  { id: 6, name: 'Forum Discussions', type: 'JSON', size: '12.8 MB', rows: 3400, created: '1 week ago', status: 'Processing' },
];

export default function Datasets() {
  return (
    <div className="space-y-8 max-w-[1600px] mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center">
            <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Datasets</h1>
                <p className="text-gray-500 dark:text-gray-400 mt-1">Manage and export generated data</p>
            </div>
            <div className="flex gap-3">
                <div className="relative">
                    <Search className="absolute left-3 top-2.5 w-4 h-4 text-gray-400" />
                    <input 
                        type="text" 
                        placeholder="Search datasets..." 
                        className="bg-white dark:bg-[#15171C] border border-gray-200 dark:border-white/10 rounded-xl py-2 pl-10 pr-4 text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500/50 w-64"
                    />
                </div>
                <button className="flex items-center gap-2 px-4 py-2 rounded-xl border border-gray-200 dark:border-white/10 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-white/5 transition-colors font-medium bg-white dark:bg-[#15171C]">
                    <Filter className="w-4 h-4" />
                    Filter
                </button>
            </div>
        </div>

        {/* Datasets Table */}
        <div className="bg-white dark:bg-[#15171C] rounded-3xl border border-gray-100 dark:border-white/5 shadow-sm overflow-hidden">
            <div className="overflow-x-auto">
                <table className="w-full">
                    <thead>
                        <tr className="border-b border-gray-100 dark:border-white/5 bg-gray-50/50 dark:bg-white/[0.02]">
                            <th className="text-left py-4 px-6 text-xs font-semibold text-gray-500 uppercase tracking-wider">Name</th>
                            <th className="text-left py-4 px-6 text-xs font-semibold text-gray-500 uppercase tracking-wider">Type</th>
                            <th className="text-left py-4 px-6 text-xs font-semibold text-gray-500 uppercase tracking-wider">Size</th>
                            <th className="text-left py-4 px-6 text-xs font-semibold text-gray-500 uppercase tracking-wider">Rows</th>
                            <th className="text-left py-4 px-6 text-xs font-semibold text-gray-500 uppercase tracking-wider">Created</th>
                            <th className="text-left py-4 px-6 text-xs font-semibold text-gray-500 uppercase tracking-wider">Status</th>
                            <th className="text-right py-4 px-6 text-xs font-semibold text-gray-500 uppercase tracking-wider">Actions</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 dark:divide-white/5">
                        {datasets.map((dataset) => (
                            <tr key={dataset.id} className="group hover:bg-gray-50 dark:hover:bg-white/[0.02] transition-colors">
                                <td className="py-4 px-6">
                                    <div className="flex items-center gap-3">
                                        <div className="p-2 rounded-lg bg-indigo-50 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">
                                            {dataset.type === 'JSON' ? <FileJson className="w-4 h-4" /> : <FileText className="w-4 h-4" />}
                                        </div>
                                        <span className="font-medium text-gray-900 dark:text-white">{dataset.name}</span>
                                    </div>
                                </td>
                                <td className="py-4 px-6">
                                    <span className="text-sm font-mono text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-white/5 px-2 py-1 rounded">{dataset.type}</span>
                                </td>
                                <td className="py-4 px-6 text-sm text-gray-500 dark:text-gray-400">{dataset.size}</td>
                                <td className="py-4 px-6 text-sm text-gray-500 dark:text-gray-400">{dataset.rows}</td>
                                <td className="py-4 px-6 text-sm text-gray-500 dark:text-gray-400">{dataset.created}</td>
                                <td className="py-4 px-6">
                                    <span className={cn(
                                        "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium",
                                        dataset.status === 'Ready' ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-500/10 dark:text-emerald-400" :
                                        dataset.status === 'Processing' ? "bg-indigo-100 text-indigo-800 dark:bg-indigo-500/10 dark:text-indigo-400" :
                                        "bg-gray-100 text-gray-800 dark:bg-gray-500/10 dark:text-gray-400"
                                    )}>
                                        {dataset.status}
                                    </span>
                                </td>
                                <td className="py-4 px-6 text-right">
                                    <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                        <button className="p-2 text-gray-400 hover:text-indigo-500 hover:bg-indigo-50 dark:hover:bg-indigo-500/10 rounded-lg transition-colors" title="View">
                                            <Eye className="w-4 h-4" />
                                        </button>
                                        <button className="p-2 text-gray-400 hover:text-emerald-500 hover:bg-emerald-50 dark:hover:bg-emerald-500/10 rounded-lg transition-colors" title="Download">
                                            <Download className="w-4 h-4" />
                                        </button>
                                        <button className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-500/10 rounded-lg transition-colors" title="Delete">
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            
            {/* Pagination */}
            <div className="px-6 py-4 border-t border-gray-100 dark:border-white/5 flex items-center justify-between">
                <div className="text-sm text-gray-500 dark:text-gray-400">
                    Showing <span className="font-medium text-gray-900 dark:text-white">1</span> to <span className="font-medium text-gray-900 dark:text-white">6</span> of <span className="font-medium text-gray-900 dark:text-white">12</span> results
                </div>
                <div className="flex gap-2">
                    <button className="px-3 py-1 text-sm rounded-lg border border-gray-200 dark:border-white/10 text-gray-500 disabled:opacity-50 hover:bg-gray-50 dark:hover:bg-white/5 transition-colors">Previous</button>
                    <button className="px-3 py-1 text-sm rounded-lg border border-gray-200 dark:border-white/10 text-gray-500 hover:bg-gray-50 dark:hover:bg-white/5 transition-colors">Next</button>
                </div>
            </div>
        </div>
    </div>
  );
}
