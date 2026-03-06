import React, { useState } from 'react';
import { Mail, Lock, Eye, EyeOff, Loader2, ArrowRight, Github, Chrome } from 'lucide-react';
import { cn } from '@/lib/utils';
import { api } from '@/services/api';

export default function Login({ 
  onSignupClick, 
  onLoginSuccess 
}: { 
  onSignupClick: () => void,
  onLoginSuccess: () => void 
}) {
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.login({ 
        username: formData.email, 
        password: formData.password 
      });
      
      localStorage.setItem('token', data.token);
      onLoginSuccess();
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred');
      console.error('Login error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-zinc-50 dark:bg-[#09090b] p-4 font-sans text-zinc-900 dark:text-zinc-100 transition-colors duration-300">
      <div className="w-full max-w-[480px] space-y-8 animate-in fade-in zoom-in duration-500">
        {/* Logo & Header */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-3xl bg-gradient-to-br from-violet-600 to-indigo-600 shadow-xl shadow-violet-500/20 mb-4 transform hover:scale-105 transition-transform duration-300">
            <svg className="w-8 h-8 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
              <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
              <line x1="12" y1="22.08" x2="12" y2="12" />
            </svg>
          </div>
          <h1 className="text-3xl font-bold text-zinc-900 dark:text-white tracking-tight">Welcome back</h1>
          <p className="text-zinc-500 dark:text-zinc-400">Log in to manage your QA pipeline</p>
        </div>

        {/* Card */}
        <div className="bg-white dark:bg-[#15171C] rounded-[32px] border border-zinc-200 dark:border-white/5 p-8 shadow-2xl shadow-zinc-200/50 dark:shadow-none">
          {error && (
            <div className="mb-6 p-4 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-600 dark:text-red-400 text-sm font-medium animate-in slide-in-from-top-2 duration-300">
              {error}
            </div>
          )}
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Email */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider ml-1">Email Address</label>
              <div className="relative">
                <Mail className="absolute left-4 top-3.5 w-4 h-4 text-zinc-400" />
                <input 
                  type="email" 
                  required
                  placeholder="name@company.com"
                  className={cn(
                    "w-full bg-zinc-50 dark:bg-white/5 border border-zinc-200 dark:border-white/10 rounded-2xl py-3.5 pl-11 pr-4 text-sm text-zinc-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50 transition-all placeholder:text-zinc-400",
                    formData.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email) && "border-red-500/50 focus:ring-red-500/20"
                  )}
                  value={formData.email}
                  onChange={(e) => setFormData({...formData, email: e.target.value})}
                />
              </div>
            </div>

            {/* Password */}
            <div className="space-y-2">
              <div className="flex justify-between items-center ml-1">
                <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Password</label>
                <button type="button" className="text-[10px] text-violet-600 font-bold hover:underline">Forgot password?</button>
              </div>
              <div className="relative">
                <Lock className="absolute left-4 top-3.5 w-4 h-4 text-zinc-400" />
                <input 
                  type={showPassword ? "text" : "password"} 
                  required
                  placeholder="••••••••"
                  className="w-full bg-zinc-50 dark:bg-white/5 border border-zinc-200 dark:border-white/10 rounded-2xl py-3.5 pl-11 pr-12 text-sm text-zinc-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-violet-500/50 transition-all placeholder:text-zinc-400"
                  value={formData.password}
                  onChange={(e) => setFormData({...formData, password: e.target.value})}
                />
                <button 
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  className="absolute right-4 top-3.5 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Submit Button */}
            <button 
              type="submit"
              disabled={isLoading}
              className="w-full flex items-center justify-center gap-2 bg-violet-600 hover:bg-violet-700 text-white py-4 rounded-2xl font-bold shadow-lg shadow-violet-500/20 transition-all active:scale-[0.98] disabled:opacity-70"
            >
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <>
                  Sign In
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>

            {/* Social Divider */}
            <div className="relative py-2">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-zinc-100 dark:border-white/5"></div>
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-white dark:bg-[#15171C] px-4 text-zinc-400 font-medium tracking-widest">Or continue with</span>
              </div>
            </div>

            {/* Social Buttons */}
            <div className="grid grid-cols-2 gap-4">
              <button type="button" className="flex items-center justify-center gap-2 py-3 px-4 rounded-2xl border border-zinc-200 dark:border-white/10 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                <Github className="w-4 h-4" /> Github
              </button>
              <button type="button" className="flex items-center justify-center gap-2 py-3 px-4 rounded-2xl border border-zinc-200 dark:border-white/10 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                <Chrome className="w-4 h-4" /> Google
              </button>
            </div>
          </form>
        </div>

        {/* Footer Link */}
        <p className="text-center text-sm text-zinc-500 dark:text-zinc-400">
          Don't have an account?{' '}
          <button 
            onClick={onSignupClick}
            className="text-violet-600 dark:text-violet-400 font-bold hover:underline underline-offset-4"
          >
            Create account
          </button>
        </p>
      </div>
    </div>
  );
}
