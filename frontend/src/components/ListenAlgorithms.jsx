/**
 * LISTEN Algorithms Component
 * Implements LISTEN-U (Utility Refinement) and LISTEN-T (Tournament Selection)
 */
import React, { useState } from 'react';
import { runListenU, runListenT } from '../api/flights';
import { format } from 'date-fns';

const ListenAlgorithms = ({ flights, userId = 'user_001' }) => {
  const [activeAlgorithm, setActiveAlgorithm] = useState(null); // 'listen-u' or 'listen-t'
  const [preferenceUtterance, setPreferenceUtterance] = useState(
    'I want to minimize price, but not if it means long layovers. I prefer flights arriving before 5 PM and ideally nonstop.'
  );
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  // LISTEN-U specific state
  const [maxIterations, setMaxIterations] = useState(3);

  // LISTEN-T specific state
  const [numRounds, setNumRounds] = useState(3);
  const [batchSize, setBatchSize] = useState(4);

  const runAlgorithm = async (algorithm) => {
    setError(null);
    setLoading(true);
    setResults(null);

    try {
      const flightIds = flights.map((f) => f.id);

      let response;
      if (algorithm === 'listen-u') {
        response = await runListenU({
          user_id: userId,
          flight_ids: flightIds,
          preference_utterance: preferenceUtterance,
          max_iterations: maxIterations,
        });
      } else if (algorithm === 'listen-t') {
        response = await runListenT({
          user_id: userId,
          flight_ids: flightIds,
          preference_utterance: preferenceUtterance,
          num_rounds: numRounds,
          batch_size: batchSize,
        });
      }

      setResults(response.results);
      setActiveAlgorithm(algorithm);
    } catch (err) {
      console.error('Algorithm error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to run algorithm');
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (dateTimeString) => {
    try {
      const date = new Date(dateTimeString);
      return format(date, 'MMM dd, HH:mm');
    } catch {
      return dateTimeString;
    }
  };

  const formatDuration = (durationMin) => {
    if (!durationMin) return 'N/A';
    const hours = Math.floor(durationMin / 60);
    const minutes = Math.round(durationMin % 60);
    return `${hours}h ${minutes}m`;
  };

  if (!flights || flights.length === 0) {
    return (
      <div className="card">
        <p className="text-gray-500">Please select flights to rank using LISTEN algorithms.</p>
      </div>
    );
  }

  // Initial selection screen
  if (!activeAlgorithm && !results) {
    return (
      <div className="card">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">LISTEN Algorithms</h2>

        <p className="text-gray-600 mb-6">
          Choose a LISTEN algorithm to automatically rank flights based on your preferences.
        </p>

        {/* Preference Utterance */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Your Preferences (Natural Language)
          </label>
          <textarea
            value={preferenceUtterance}
            onChange={(e) => setPreferenceUtterance(e.target.value)}
            rows={3}
            className="input-field w-full"
            placeholder="Describe what you're looking for in a flight..."
          />
          <p className="text-xs text-gray-500 mt-1">
            Example: "I want to minimize price, but not if it means long layovers. I prefer nonstop flights arriving before 5 PM."
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* LISTEN-U Card */}
          <div className="border-2 border-blue-200 rounded-lg p-6 hover:border-blue-400 transition-colors">
            <h3 className="text-xl font-semibold text-blue-700 mb-3">
              🎯 LISTEN-U
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              <strong>Utility Refinement</strong><br />
              Iteratively refines a linear utility function over numerical attributes
              (price, duration, stops, etc.) to find the best match.
            </p>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Iterations
              </label>
              <input
                type="number"
                value={maxIterations}
                onChange={(e) => setMaxIterations(Number(e.target.value))}
                min={1}
                max={10}
                className="input-field w-full"
              />
            </div>

            <button
              onClick={() => runAlgorithm('listen-u')}
              disabled={loading}
              className="btn-primary w-full"
            >
              {loading ? 'Running...' : 'Run LISTEN-U'}
            </button>
          </div>

          {/* LISTEN-T Card */}
          <div className="border-2 border-purple-200 rounded-lg p-6 hover:border-purple-400 transition-colors">
            <h3 className="text-xl font-semibold text-purple-700 mb-3">
              🏆 LISTEN-T
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              <strong>Tournament Selection</strong><br />
              Conducts a tournament by sampling random batches, selecting champions,
              then running a final playoff to find the winner.
            </p>

            <div className="grid grid-cols-2 gap-3 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Rounds
                </label>
                <input
                  type="number"
                  value={numRounds}
                  onChange={(e) => setNumRounds(Number(e.target.value))}
                  min={1}
                  max={10}
                  className="input-field w-full"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Batch Size
                </label>
                <input
                  type="number"
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  min={2}
                  max={10}
                  className="input-field w-full"
                />
              </div>
            </div>

            <button
              onClick={() => runAlgorithm('listen-t')}
              disabled={loading}
              className="btn-primary w-full bg-purple-600 hover:bg-purple-700"
            >
              {loading ? 'Running...' : 'Run LISTEN-T'}
            </button>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}

        <div className="bg-gray-50 p-4 rounded-lg">
          <p className="text-sm text-gray-600">
            <strong>Selected Flights:</strong> {flights.length}<br />
            These algorithms will analyze and rank your selected flights based on your preference description.
          </p>
        </div>
      </div>
    );
  }

  // Results screen
  return (
    <div className="card">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800">
          {activeAlgorithm === 'listen-u' ? '🎯 LISTEN-U Results' : '🏆 LISTEN-T Results'}
        </h2>
        <button
          onClick={() => {
            setActiveAlgorithm(null);
            setResults(null);
          }}
          className="btn-secondary"
        >
          ← Run Another Algorithm
        </button>
      </div>

      {/* Preference Utterance */}
      <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg mb-6">
        <p className="text-sm font-medium text-blue-900 mb-1">Your Preferences:</p>
        <p className="text-blue-800">"{results.preference_utterance}"</p>
      </div>

      {/* LISTEN-U Specific: Show Weights */}
      {activeAlgorithm === 'listen-u' && results.final_weights && (
        <div className="bg-green-50 border border-green-200 p-4 rounded-lg mb-6">
          <h3 className="text-lg font-semibold text-green-900 mb-3">Final Weight Vector</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {Object.entries(results.final_weights).map(([attr, weight]) => (
              <div key={attr} className="bg-white p-3 rounded">
                <p className="text-xs text-gray-600">{attr.replace('_', ' ')}</p>
                <p className="text-lg font-bold text-green-700">{(weight * 100).toFixed(1)}%</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* LISTEN-T Specific: Show Tournament Info */}
      {activeAlgorithm === 'listen-t' && results.tournament_log && (
        <div className="bg-purple-50 border border-purple-200 p-4 rounded-lg mb-6">
          <h3 className="text-lg font-semibold text-purple-900 mb-3">Tournament Summary</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-white p-3 rounded text-center">
              <p className="text-xs text-gray-600">Rounds</p>
              <p className="text-2xl font-bold text-purple-700">{results.tournament_rounds}</p>
            </div>
            <div className="bg-white p-3 rounded text-center">
              <p className="text-xs text-gray-600">Batch Size</p>
              <p className="text-2xl font-bold text-purple-700">{results.batch_size}</p>
            </div>
            <div className="bg-white p-3 rounded text-center">
              <p className="text-xs text-gray-600">Champions</p>
              <p className="text-2xl font-bold text-purple-700">{results.champions?.length || 0}</p>
            </div>
          </div>
        </div>
      )}

      {/* Winner/Best Flight */}
      {(results.best_flight || results.winner) && (
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 border-2 border-yellow-300 p-6 rounded-lg mb-6">
          <h3 className="text-xl font-bold text-yellow-900 mb-4">
            🏆 {activeAlgorithm === 'listen-u' ? 'Best Flight' : 'Tournament Winner'}
          </h3>
          {(() => {
            const topFlight = results.best_flight || results.winner;
            return (
              <div className="bg-white p-4 rounded-lg">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold text-lg text-gray-800">{topFlight.name}</h4>
                    <p className="text-gray-600">{topFlight.origin} → {topFlight.destination}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-3xl font-bold text-green-600">${topFlight.price?.toFixed(2)}</p>
                    {topFlight.utility_score && (
                      <p className="text-sm text-gray-600">Score: {topFlight.utility_score.toFixed(3)}</p>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 mt-4 text-sm">
                  <div>
                    <p className="text-gray-600">Departure</p>
                    <p className="font-medium">{formatTime(topFlight.departure_time)}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Duration</p>
                    <p className="font-medium">{formatDuration(topFlight.duration_min)}</p>
                  </div>
                  <div>
                    <p className="text-gray-600">Stops</p>
                    <p className="font-medium">{topFlight.stops}</p>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* Ranked Flights List */}
      {results.ranked_flights && results.ranked_flights.length > 0 && (
        <div>
          <h3 className="text-xl font-semibold text-gray-800 mb-4">All Flights Ranked</h3>
          <div className="space-y-3">
            {results.ranked_flights.map((flight, index) => (
              <div
                key={flight.id}
                className={`border rounded-lg p-4 ${
                  index === 0 ? 'border-yellow-300 bg-yellow-50' : 'border-gray-200 bg-white'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4 flex-1">
                    <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                      <span className="text-sm font-bold text-blue-700">{flight.rank || index + 1}</span>
                    </div>
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-800">{flight.name}</h4>
                      <p className="text-sm text-gray-600">
                        {flight.origin} → {flight.destination} | {formatTime(flight.departure_time)}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-lg font-bold text-green-600">${flight.price?.toFixed(2)}</p>
                    {flight.utility_score && (
                      <p className="text-xs text-gray-500">Score: {flight.utility_score.toFixed(3)}</p>
                    )}
                    <p className="text-xs text-gray-600">
                      {formatDuration(flight.duration_min)} | {flight.stops} stops
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ListenAlgorithms;
