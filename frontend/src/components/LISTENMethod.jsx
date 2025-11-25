/**
 * LISTEN Method Component
 * Natural language prompt → LISTEN-U ranking → Top 5 selection
 */
import React, { useState } from 'react';
import FlightSearch from './FlightSearch';
import FlightCard from './FlightCard';
import Top5Panel from './Top5Panel';
import { rankFlightsListenU } from '../api/flights';

const LISTENMethod = ({ onComplete }) => {
  const [prompt, setPrompt] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [rankedFlights, setRankedFlights] = useState([]);
  const [topFlights, setTopFlights] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearchComplete = async (results) => {
    setSearchResults(results);
    setLoading(true);
    setError(null);

    try {
      // Get the user's prompt from the search
      const userPrompt = results.parsed_params?.user_prompt || prompt;

      // Run LISTEN-U algorithm
      const response = await rankFlightsListenU(
        results.flights,
        userPrompt,
        null
      );

      // Set ranked flights
      setRankedFlights(response.ranked_flights || results.flights);
      setPrompt(userPrompt);
    } catch (err) {
      console.error('LISTEN-U error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to run LISTEN-U');
      // Fallback to unranked flights
      setRankedFlights(results.flights);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleSelect = (flight) => {
    setTopFlights(prev => {
      const isSelected = prev.some(f => f.id === flight.id);
      if (isSelected) {
        return prev.filter(f => f.id !== flight.id);
      } else if (prev.length < 5) {
        return [...prev, { ...flight, id: flight.id || Math.random() }];
      }
      return prev;
    });
  };

  const handleReorder = (reorderedFlights) => {
    setTopFlights(reorderedFlights);
  };

  const handleRemove = (flight) => {
    setTopFlights(prev => prev.filter(f => f.id !== flight.id));
  };

  const handleSubmit = async () => {
    // TODO: Save to database via API
    console.log('LISTEN Method - Top 5 Rankings:', topFlights);
    console.log('LISTEN Method - Prompt:', prompt);

    // Call onComplete to move to next method
    if (onComplete) {
      onComplete({
        method: 'listen',
        rankings: topFlights,
        prompt: prompt,
        searchResults: searchResults,
      });
    }
  };

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="text-2xl font-bold mb-2 text-gray-800">
          Method 2: LISTEN-U Algorithm
        </h2>
        <p className="text-gray-600 mb-4">
          Enter your flight preferences in natural language. LISTEN-U will rank flights based on your preferences.
        </p>
        {loading && (
          <div className="bg-blue-50 border border-blue-200 text-blue-800 px-4 py-3 rounded">
            Running LISTEN-U algorithm... This may take 2-3 minutes.
          </div>
        )}
        {error && (
          <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 px-4 py-3 rounded">
            {error} - Showing unranked flights as fallback.
          </div>
        )}
      </div>

      {!searchResults && (
        <FlightSearch onSearchComplete={handleSearchComplete} />
      )}

      {searchResults && rankedFlights.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Flights List */}
          <div className="lg:col-span-2 space-y-4">
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-lg font-bold text-gray-800">
                    LISTEN-U Ranked Flights ({rankedFlights.length})
                  </h3>
                  <p className="text-sm text-gray-600 mt-1">
                    Flights ranked by utility (best match at top)
                  </p>
                </div>
                <button
                  onClick={() => {
                    setSearchResults(null);
                    setRankedFlights([]);
                    setTopFlights([]);
                  }}
                  className="btn-secondary text-sm"
                >
                  New Search
                </button>
              </div>

              <div className="space-y-3 max-h-[800px] overflow-y-auto">
                {rankedFlights.map((flight, index) => (
                  <div key={flight.id || index}>
                    <div className="flex items-center gap-2 mb-2">
                      <span className="bg-green-600 text-white text-xs font-bold px-2 py-1 rounded">
                        LISTEN Rank #{index + 1}
                      </span>
                    </div>
                    <FlightCard
                      flight={flight}
                      isSelected={topFlights.some(f => f.id === flight.id)}
                      onToggleSelect={handleToggleSelect}
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Top 5 Panel */}
          <div className="lg:col-span-1">
            <Top5Panel
              topFlights={topFlights}
              onReorder={handleReorder}
              onRemove={handleRemove}
              onSubmit={handleSubmit}
              methodName="LISTEN"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default LISTENMethod;
