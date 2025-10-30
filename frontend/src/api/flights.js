/**
 * API client for Flight Metrics backend
 * All API calls are handled through axios
 */
import axios from 'axios';

// Base URL for API calls
// In development, this will proxy through the React dev server to port 8000
// In production, this should be configured to point to your backend server
const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

/**
 * Flight Search and Management APIs
 */

export const searchFlights = async (params) => {
  const response = await apiClient.get('/flights/search', { params });
  return response.data;
};

export const getAllFlights = async (filters = {}) => {
  const response = await apiClient.get('/flights/all', { params: filters });
  return response.data;
};

export const getFlightMetrics = async (filters = {}) => {
  const response = await apiClient.get('/flights/metrics', { params: filters });
  return response.data;
};

export const getFlightById = async (flightId) => {
  const response = await apiClient.get(`/flights/${flightId}`);
  return response.data;
};

/**
 * LISTEN Evaluation APIs
 */

export const submitListenRanking = async (data) => {
  const response = await apiClient.post('/evaluate/listen/ranking', data);
  return response.data;
};

export const getListenRankings = async (userId = null, limit = 100) => {
  const params = { limit };
  if (userId) params.user_id = userId;
  const response = await apiClient.get('/evaluate/listen/rankings', { params });
  return response.data;
};

/**
 * Team Draft Evaluation APIs
 */

export const startTeamDraft = async (data) => {
  const response = await apiClient.post('/evaluate/teamdraft/start', data);
  return response.data;
};

export const submitTeamDraftPreferences = async (data) => {
  const response = await apiClient.post('/evaluate/teamdraft/submit', data);
  return response.data;
};

export const getTeamDraftResults = async (sessionId) => {
  const response = await apiClient.get(`/evaluate/teamdraft/results/${sessionId}`);
  return response.data;
};

/**
 * Rating APIs
 */

export const submitRating = async (data) => {
  const response = await apiClient.post('/evaluate/rating', data);
  return response.data;
};

export const getRatings = async (filters = {}) => {
  const response = await apiClient.get('/evaluate/ratings', { params: filters });
  return response.data;
};

/**
 * LISTEN Algorithm APIs
 */

export const runListenU = async (data) => {
  const response = await apiClient.post('/evaluate/listen-u/run', data);
  return response.data;
};

export const runListenT = async (data) => {
  const response = await apiClient.post('/evaluate/listen-t/run', data);
  return response.data;
};

/**
 * System APIs
 */

export const healthCheck = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};

export const getSystemInfo = async () => {
  const response = await apiClient.get('/info');
  return response.data;
};

// Export default object with all functions
const flightsAPI = {
  searchFlights,
  getAllFlights,
  getFlightMetrics,
  getFlightById,
  submitListenRanking,
  getListenRankings,
  startTeamDraft,
  submitTeamDraftPreferences,
  getTeamDraftResults,
  submitRating,
  getRatings,
  runListenU,
  runListenT,
  healthCheck,
  getSystemInfo,
};

export default flightsAPI;
