/**
 * FlightSearch Component
 * Natural language flight search with Gemini + Amadeus
 */
import React, { useState } from 'react';
import { searchFlights } from '../api/flights';

const FlightSearch = ({ onSearchComplete }) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      // Validate input
      if (!query.trim()) {
        throw new Error('Please enter your flight search query');
      }

      // Generate session ID
      const sessionId = `search_${Date.now()}_${Math.random().toString(36).substring(7)}`;

      // Call API with natural language query
      const result = await searchFlights(query, sessionId);

      // Store session ID in result for tracking
      result.sessionId = sessionId;

      // Pass results to parent component
      if (onSearchComplete) {
        onSearchComplete(result);
      }
    } catch (err) {
      console.error('Search error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to search flights');
    } finally {
      setLoading(false);
    }
  };

  // Quick example prompts
  const loadExample = (exampleQuery) => {
    setQuery(exampleQuery);
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Search Flights</h2>

      <p className="text-gray-600 mb-4">
        Describe your flight needs in natural language. Our AI will parse your request and find the best flights.
      </p>

      {/* Quick Examples */}
      <div className="mb-6">
        <p className="text-sm font-medium text-gray-700 mb-2">Try these examples:</p>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => loadExample('Find flights from JFK to LAX on December 15, 2025 for 2 adults')}
            className="btn-secondary text-sm"
          >
            JFK to LAX
          </button>
          <button
            type="button"
            onClick={() => loadExample('I need to fly from SFO to JFK next week for 1 adult')}
            className="btn-secondary text-sm"
          >
            SFO to JFK
          </button>
          <button
            type="button"
            onClick={() => loadExample('Show me flights from ORD to MIA on January 5, 2026 for 3 adults')}
            className="btn-secondary text-sm"
          >
            ORD to MIA
          </button>
          <button
            type="button"
            onClick={() => loadExample('I want to go from LAX to London LHR on December 20, 2025 for 2 people')}
            className="btn-secondary text-sm"
          >
            LAX to London
          </button>
        </div>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
            What flight are you looking for?
          </label>
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="E.g., 'I want to fly from JFK to LAX on December 15, 2025 for 2 adults' or 'Find me flights from San Francisco to New York next Monday for 1 person'"
            className="input-field w-full h-24 resize-none"
            required
          />
          <p className="text-xs text-gray-500 mt-1">
            Include: origin, destination, date, and number of adults
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}

        {/* Submit Button */}
        <div>
          <button
            type="submit"
            disabled={loading}
            className="btn-primary w-full md:w-auto px-8"
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Searching flights...
              </span>
            ) : (
              'Search Flights'
            )}
          </button>
        </div>
      </form>

      {/* How it works */}
      <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <h3 className="text-sm font-semibold text-blue-900 mb-2">How it works:</h3>
        <ol className="text-xs text-blue-800 space-y-1 list-decimal list-inside">
          <li>Gemini AI parses your natural language query</li>
          <li>Extracts flight details (origin, destination, date, passengers)</li>
          <li>Searches real flights via Amadeus API</li>
          <li>Returns results you can rank with LISTEN-U or LILO</li>
        </ol>
      </div>
    </div>
  );
};

export default FlightSearch;
