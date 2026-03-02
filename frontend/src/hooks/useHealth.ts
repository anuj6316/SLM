import { useState } from 'react';
import { api } from '@/services/api';
import { HealthRequest, HealthResponse } from '@/types';

export function useHealth() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<HealthResponse | null>(null);

  const checkStatus = async (keys: HealthRequest) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await api.checkHealth(keys);
      setData(result);
      return result;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(message);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  return { 
    checkStatus, 
    isLoading, 
    error, 
    data,
    groqActive: data?.groq_isActive ?? false,
    jinaActive: data?.jina_isActive ?? false
  };
}
