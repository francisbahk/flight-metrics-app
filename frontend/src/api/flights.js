/**
 * API client for Flight Metrics backend (FastAPI)
 * All API calls are handled through axios
 */
import axios from 'axios';

// Base URL for API calls
// In development, proxy is configured in package.json to port 8000
// In production, set REACT_APP_API_URL environment variable
const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 180000, // 3 minute timeout (LISTEN-U takes 2-3 minutes)
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
 * Flight Search API
 */

export const searchFlights = async (query, sessionId = null) => {
  const response = await apiClient.post('/search', {
    query,
    session_id: sessionId,
  });
  return response.data;
};

/**
 * Ranking APIs
 */

export const rankFlightsCheapest = async (flights, userPrompt, userPreferences = null) => {
  const response = await apiClient.post('/rank/cheapest', {
    flights,
    user_prompt: userPrompt,
    user_preferences: userPreferences,
  });
  return response.data;
};

export const rankFlightsFastest = async (flights, userPrompt, userPreferences = null) => {
  const response = await apiClient.post('/rank/fastest', {
    flights,
    user_prompt: userPrompt,
    user_preferences: userPreferences,
  });
  return response.data;
};

export const rankFlightsListenU = async (flights, userPrompt, userPreferences = null) => {
  const response = await apiClient.post('/rank/listen-u', {
    flights,
    user_prompt: userPrompt,
    user_preferences: userPreferences,
  });
  return response.data;
};

export const rankFlightsAll = async (flights, userPrompt, userPreferences = null) => {
  const response = await apiClient.post('/rank/all', {
    flights,
    user_prompt: userPrompt,
    user_preferences: userPreferences,
  });
  return response.data;
};

/**
 * LILO APIs (2-round interactive workflow)
 */

export const initLilo = async (sessionId, flights, userPrompt, userPreferences = null) => {
  const response = await apiClient.post('/lilo/init', {
    session_id: sessionId,
    flights,
    user_prompt: userPrompt,
    user_preferences: userPreferences,
  });
  return response.data;
};

export const submitLiloRound = async (sessionId, roundNumber, userRankings, userFeedback) => {
  const response = await apiClient.post('/lilo/round', {
    session_id: sessionId,
    round_number: roundNumber,
    user_rankings: userRankings,
    user_feedback: userFeedback,
  });
  return response.data;
};

export const getLiloFinal = async (sessionId) => {
  const response = await apiClient.post('/lilo/final', {
    session_id: sessionId,
  });
  return response.data;
};

// New Gemini-powered LILO functions
export const generateLiloQuestions = async (userPrompt) => {
  const response = await apiClient.post('/lilo/generate-questions', {
    user_prompt: userPrompt,
  });
  return response.data;
};

export const rankWithFeedback = async (sessionId, flights, feedback, initialAnswers = {}) => {
  const response = await apiClient.post('/lilo/rank-with-feedback', {
    session_id: sessionId,
    flights,
    feedback,
    initial_answers: initialAnswers,
  });
  return response.data;
};

/**
 * Evaluation APIs (Person A vs Person B vs Algorithms)
 */

export const startEvaluationSession = async (sessionId, userId, prompt) => {
  const response = await apiClient.post('/evaluation/start', {
    session_id: sessionId,
    user_id: userId,
    prompt,
  });
  return response.data;
};

export const submitPersonARankings = async (sessionId, rankings) => {
  const response = await apiClient.post('/evaluation/person-a/rankings', {
    session_id: sessionId,
    rankings,
  });
  return response.data;
};

export const submitPersonBRankings = async (evalSessionId, userId, rankings) => {
  const response = await apiClient.post('/evaluation/person-b/rankings', {
    eval_session_id: evalSessionId,
    user_id: userId,
    rankings,
  });
  return response.data;
};

export const submitAlgorithmRankings = async (evalSessionId, algorithm, rankings) => {
  const response = await apiClient.post(`/evaluation/algorithm/rankings/${algorithm}`, {
    eval_session_id: evalSessionId,
    rankings,
  });
  return response.data;
};

export const getEvaluationSession = async (sessionId) => {
  const response = await apiClient.get(`/evaluation/session/${sessionId}`);
  return response.data;
};

export const compareRankings = async (evalSessionId) => {
  const response = await apiClient.post('/evaluation/compare', {
    eval_session_id: evalSessionId,
  });
  return response.data;
};

/**
 * Tracking APIs (interaction events)
 */

export const trackEvent = async (sessionId, eventType, eventData, searchId = null) => {
  const response = await apiClient.post('/tracking/event', {
    session_id: sessionId,
    search_id: searchId,
    event_type: eventType,
    event_data: eventData,
  });
  return response.data;
};

export const getSessionEvents = async (sessionId) => {
  const response = await apiClient.get(`/tracking/events/${sessionId}`);
  return response.data;
};

export const getSearchAnalytics = async (searchId) => {
  const response = await apiClient.get(`/tracking/analytics/${searchId}`);
  return response.data;
};

/**
 * Sequential Evaluation APIs (Manual → LISTEN → LILO)
 */

export const submitManualEvaluation = async (sessionId, userId, searchResults, rankings) => {
  const response = await apiClient.post('/evaluation/sequential/manual', {
    session_id: sessionId,
    user_id: userId,
    search_results: searchResults,
    rankings,
  });
  return response.data;
};

export const submitLISTENEvaluation = async (sessionId, prompt, searchResults, rankedFlights, rankings) => {
  const response = await apiClient.post('/evaluation/sequential/listen', {
    session_id: sessionId,
    prompt,
    search_results: searchResults,
    ranked_flights: rankedFlights,
    rankings,
  });
  return response.data;
};

export const submitLILOEvaluation = async (
  sessionId,
  prompt,
  searchResults,
  initialAnswers,
  iteration1Flights,
  iteration1Feedback,
  iteration2Flights,
  iteration2Feedback,
  iteration3Flights,
  rankings
) => {
  const response = await apiClient.post('/evaluation/sequential/lilo', {
    session_id: sessionId,
    prompt,
    search_results: searchResults,
    initial_answers: initialAnswers,
    iteration1_flights: iteration1Flights,
    iteration1_feedback: iteration1Feedback,
    iteration2_flights: iteration2Flights,
    iteration2_feedback: iteration2Feedback,
    iteration3_flights: iteration3Flights,
    rankings,
  });
  return response.data;
};

export const getSequentialEvaluation = async (sessionId) => {
  const response = await apiClient.get(`/evaluation/sequential/${sessionId}`);
  return response.data;
};

/**
 * System APIs
 */

export const healthCheck = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};

// Export default object with all functions
const flightsAPI = {
  // Search
  searchFlights,

  // Ranking
  rankFlightsCheapest,
  rankFlightsFastest,
  rankFlightsListenU,
  rankFlightsAll,

  // LILO
  initLilo,
  submitLiloRound,
  getLiloFinal,

  // Evaluation
  startEvaluationSession,
  submitPersonARankings,
  submitPersonBRankings,
  submitAlgorithmRankings,
  getEvaluationSession,
  compareRankings,

  // Sequential Evaluation
  submitManualEvaluation,
  submitLISTENEvaluation,
  submitLILOEvaluation,
  getSequentialEvaluation,

  // Tracking
  trackEvent,
  getSessionEvents,
  getSearchAnalytics,

  // System
  healthCheck,
};

export default flightsAPI;
