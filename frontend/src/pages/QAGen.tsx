import React, { useState } from 'react';
import { Bot, Sparkles, RefreshCw, MessageSquare, ArrowRight, Zap } from 'lucide-react';
import { cn } from '@/lib/utils';

export default function QAGen() {
  const [prompt, setPrompt] = useState("Generate 5 QA pairs from the following text. Focus on factual accuracy and technical details.");
  
  return (
    <div className="space-y-8 max-w-[1600px] mx-auto h-[calc(100vh-8rem)] flex flex-col">
        {/* Header */}
        <div className="flex justify-between items-center flex-shrink-0">
            <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">QA Generation</h1>
                <p className="text-gray-500 dark:text-gray-400 mt-1">Configure LLM parameters and preview output</p>
            </div>
            <div className="flex gap-3">
                <button className="flex items-center gap-2 px-4 py-2 rounded-xl border border-gray-200 dark:border-white/10 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-white/5 transition-colors font-medium">
                    <RefreshCw className="w-4 h-4" />
                    Reset
                </button>
                <button className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-5 py-2 rounded-xl font-medium transition-all shadow-lg shadow-indigo-500/20 active:scale-95">
                    <Sparkles className="w-4 h-4" />
                    Generate Preview
                </button>
            </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 flex-1 min-h-0">
            
            {/* Left Column: Configuration */}
            <div className="lg:col-span-4 flex flex-col gap-6 overflow-y-auto pr-2">
                
                {/* Model Settings */}
                <div className="bg-white dark:bg-[#15171C] rounded-3xl border border-gray-100 dark:border-white/5 p-6 shadow-sm">
                    <div className="flex items-center gap-3 mb-6">
                        <div className="p-2 bg-indigo-100 dark:bg-indigo-500/10 rounded-lg text-indigo-600 dark:text-indigo-400">
                            <Bot className="w-5 h-5" />
                        </div>
                        <h2 className="text-lg font-bold text-gray-900 dark:text-white">Model Config</h2>
                    </div>
                    
                    <div className="space-y-5">
                        <div className="space-y-2">
                            <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Provider</label>
                            <select className="w-full bg-gray-50 dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-xl py-3 px-4 text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500/50 appearance-none cursor-pointer">
                                <option>Groq (Llama 3 70B)</option>
                                <option>OpenAI (GPT-4o)</option>
                                <option>Anthropic (Claude 3.5 Sonnet)</option>
                            </select>
                        </div>

                        <div className="space-y-2">
                            <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Chunk Size (Tokens)</label>
                            <div className="flex items-center gap-4">
                                <input 
                                    type="range" 
                                    min="256" 
                                    max="4096" 
                                    step="256" 
                                    defaultValue="1024"
                                    className="flex-1 h-2 bg-gray-200 dark:bg-white/10 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                                />
                                <span className="text-sm font-mono text-gray-600 dark:text-gray-300 w-12 text-right">1024</span>
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Temperature</label>
                            <div className="flex items-center gap-4">
                                <input 
                                    type="range" 
                                    min="0" 
                                    max="1" 
                                    step="0.1" 
                                    defaultValue="0.3"
                                    className="flex-1 h-2 bg-gray-200 dark:bg-white/10 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                                />
                                <span className="text-sm font-mono text-gray-600 dark:text-gray-300 w-12 text-right">0.3</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Prompt Template */}
                <div className="bg-white dark:bg-[#15171C] rounded-3xl border border-gray-100 dark:border-white/5 p-6 shadow-sm flex-1 flex flex-col">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 bg-amber-100 dark:bg-amber-500/10 rounded-lg text-amber-600 dark:text-amber-400">
                            <MessageSquare className="w-5 h-5" />
                        </div>
                        <h2 className="text-lg font-bold text-gray-900 dark:text-white">System Prompt</h2>
                    </div>
                    <textarea 
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        className="flex-1 w-full bg-gray-50 dark:bg-white/5 border border-gray-200 dark:border-white/10 rounded-xl p-4 text-sm font-mono text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500/50 resize-none leading-relaxed"
                    />
                </div>
            </div>

            {/* Right Column: Preview */}
            <div className="lg:col-span-8 bg-[#1e1e1e] rounded-3xl border border-gray-800 shadow-2xl flex flex-col overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-800 bg-[#252526] flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="flex gap-1.5">
                            <div className="w-3 h-3 rounded-full bg-[#FF5F56]" />
                            <div className="w-3 h-3 rounded-full bg-[#FFBD2E]" />
                            <div className="w-3 h-3 rounded-full bg-[#27C93F]" />
                        </div>
                        <span className="text-xs font-mono text-gray-400 ml-2">Preview Output</span>
                    </div>
                    <div className="flex items-center gap-2 px-2 py-1 rounded bg-indigo-500/20 text-indigo-400 text-[10px] font-mono uppercase">
                        <Zap className="w-3 h-3" />
                        Groq-Llama3-70b
                    </div>
                </div>

                <div className="flex-1 p-8 overflow-y-auto">
                    <div className="space-y-8">
                        {/* Sample QA Pair 1 */}
                        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                            <div className="flex gap-4">
                                <div className="w-8 h-8 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-xs flex-shrink-0">Q</div>
                                <div className="text-gray-300 leading-relaxed">
                                    What is the primary function of the ingestion layer in the data pipeline?
                                </div>
                            </div>
                            <div className="flex gap-4">
                                <div className="w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-xs flex-shrink-0">A</div>
                                <div className="text-gray-400 leading-relaxed">
                                    The ingestion layer is responsible for collecting raw data from various sources, such as web scraping targets or document uploads, and normalizing it into a standard format for downstream processing.
                                </div>
                            </div>
                        </div>

                        <div className="w-full h-px bg-gray-800" />

                        {/* Sample QA Pair 2 */}
                        <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500 delay-100">
                            <div className="flex gap-4">
                                <div className="w-8 h-8 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-xs flex-shrink-0">Q</div>
                                <div className="text-gray-300 leading-relaxed">
                                    How does the system handle rate limiting during the scraping process?
                                </div>
                            </div>
                            <div className="flex gap-4">
                                <div className="w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold text-xs flex-shrink-0">A</div>
                                <div className="text-gray-400 leading-relaxed">
                                    The system implements a concurrency controller that manages the number of active requests. It also respects `robots.txt` directives and includes automatic backoff strategies when 429 (Too Many Requests) errors are encountered.
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
  );
}
