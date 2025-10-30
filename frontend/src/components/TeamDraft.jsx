/**
 * TeamDraft Component
 * Interleaved ranking evaluation - present flights one at a time for binary preference
 */
import React, { useState } from 'react';
import { startTeamDraft, submitTeamDraftPreferences } from '../api/flights';
import { format } from 'date-fns';

const TeamDraft = ({
  selectedFlights,
  userId = 'user_001',
  onComplete,
}) => {
  const [sessionId, setSessionId] = useState(null);
  const [interleavedFlights, setInterleavedFlights] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [preferences, setPreferences] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  // Algorithm settings
  const [algorithmA, setAlgorithmA] = useState('Price-Based');
  const [algorithmB, setAlgorithmB] = useState('Duration-Based');

  // Initialize Team Draft session
  const initializeSession = async () => {
    setError(null);
    setLoading(true);

    try {
      // Split flights into two algorithm rankings
      // Algorithm A: Sort by price (ascending)
      const rankingA = [...selectedFlights]
        .sort((a, b) => a.price - b.price)
        .map((f) => f.id);

      // Algorithm B: Sort by duration (ascending)
      const rankingB = [...selectedFlights]
        .sort((a, b) => (a.duration_min || 0) - (b.duration_min || 0))
        .map((f) => f.id);

      // Start Team Draft session
      const response = await startTeamDraft({
        user_id: userId,
        prompt: `Compare ${algorithmA} vs ${algorithmB}`,
        algorithm_a: algorithmA,
        algorithm_b: algorithmB,
        algorithm_a_ranking: rankingA,
        algorithm_b_ranking: rankingB,
      });

      setSessionId(response.session_id);
      setInterleavedFlights(response.flights_data);
      setCurrentIndex(0);
      setPreferences([]);
      setResults(null);
    } catch (err) {
      console.error('Initialization error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to start Team Draft');
    } finally {
      setLoading(false);
    }
  };

  // Handle preference selection
  const handlePreference = (liked) => {
    const newPreferences = [...preferences, liked];
    setPreferences(newPreferences);

    // Move to next flight
    if (currentIndex < interleavedFlights.length - 1) {
      setCurrentIndex(currentIndex + 1);
    } else {
      // All flights evaluated, submit results
      submitResults(newPreferences);
    }
  };

  // Submit final results
  const submitResults = async (finalPreferences) => {
    setLoading(true);

    try {
      const response = await submitTeamDraftPreferences({
        session_id: sessionId,
        user_preferences: finalPreferences,
      });

      setResults(response.results);

      // Notify parent component
      if (onComplete) {
        onComplete(response);
      }
    } catch (err) {
      console.error('Submit error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to submit preferences');
    } finally {
      setLoading(false);
    }
  };

  // Format time
  const formatTime = (dateTimeString) => {
    try {
      const date = new Date(dateTimeString);
      return format(date, 'MMM dd, HH:mm');
    } catch {
      return dateTimeString;
    }
  };

  // Format duration
  const formatDuration = (durationMin) => {
    if (!durationMin) return 'N/A';
    const hours = Math.floor(durationMin / 60);
    const minutes = Math.round(durationMin % 60);
    return `${hours}h ${minutes}m`;
  };

  // Show initialization screen
  if (!sessionId) {
    return (
      <div className="card">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">Team Draft Evaluation</h2>

        <p className="text-gray-600 mb-6">
          Team Draft presents flights from two different ranking algorithms in an interleaved
          manner. You'll see one flight at a time and indicate whether you like it or not.
          This helps determine which algorithm produces better results.
        </p>

        {/* Algorithm Selection */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Algorithm A
            </label>
            <input
              type="text"
              value={algorithmA}
              onChange={(e) => setAlgorithmA(e.target.value)}
              className="input-field w-full"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Algorithm B
            </label>
            <input
              type="text"
              value={algorithmB}
              onChange={(e) => setAlgorithmB(e.target.value)}
              className="input-field w-full"
            />
          </div>
        </div>

        <div className="bg-blue-50 border border-blue-200 p-4 rounded-lg mb-6">
          <p className="text-sm text-blue-800">
            <strong>Selected Flights:</strong> {selectedFlights.length}
          </p>
          <p className="text-sm text-blue-800 mt-1">
            You will evaluate {selectedFlights.length * 2} flights total
          </p>
        </div>

        {error && (
          <div className="mb-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}

        <button
          onClick={initializeSession}
          disabled={loading || selectedFlights.length === 0}
          className="btn-primary w-full md:w-auto px-8"
        >
          {loading ? 'Starting...' : 'Start Team Draft'}
        </button>
      </div>
    );
  }

  // Show results screen
  if (results) {
    return (
      <div className="card">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">Team Draft Results</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          {/* Algorithm A Score */}
          <div className={`p-6 rounded-lg ${results.winner === algorithmA ? 'bg-green-100 border-2 border-green-500' : 'bg-gray-100'}`}>
            <h3 className="text-lg font-semibold mb-2">{results.algorithm_a}</h3>
            <div className="text-4xl font-bold text-green-600">
              {results.a_score}
            </div>
            <p className="text-sm text-gray-600 mt-1">liked flights</p>
            {results.winner === algorithmA && (
              <span className="inline-block mt-2 px-3 py-1 bg-green-500 text-white rounded-full text-sm font-semibold">
                Winner!
              </span>
            )}
          </div>

          {/* Algorithm B Score */}
          <div className={`p-6 rounded-lg ${results.winner === algorithmB ? 'bg-green-100 border-2 border-green-500' : 'bg-gray-100'}`}>
            <h3 className="text-lg font-semibold mb-2">{results.algorithm_b}</h3>
            <div className="text-4xl font-bold text-blue-600">
              {results.b_score}
            </div>
            <p className="text-sm text-gray-600 mt-1">liked flights</p>
            {results.winner === algorithmB && (
              <span className="inline-block mt-2 px-3 py-1 bg-green-500 text-white rounded-full text-sm font-semibold">
                Winner!
              </span>
            )}
          </div>
        </div>

        {results.winner === 'tie' && (
          <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg mb-6">
            <p className="text-yellow-800 font-medium">
              It's a tie! Both algorithms performed equally well.
            </p>
          </div>
        )}

        <div className="bg-gray-50 p-4 rounded-lg">
          <h4 className="font-semibold mb-2">Summary</h4>
          <ul className="text-sm text-gray-600 space-y-1">
            <li>Total flights evaluated: {interleavedFlights.length}</li>
            <li>Flights you liked: {results.total_yes}</li>
            <li>Flights you disliked: {results.total_no}</li>
          </ul>
        </div>

        <button
          onClick={() => {
            setSessionId(null);
            setInterleavedFlights([]);
            setCurrentIndex(0);
            setPreferences([]);
            setResults(null);
          }}
          className="btn-secondary mt-6"
        >
          Start New Evaluation
        </button>
      </div>
    );
  }

  // Show evaluation screen
  const currentFlight = interleavedFlights[currentIndex]?.flight;
  const progress = ((currentIndex + 1) / interleavedFlights.length) * 100;

  if (!currentFlight) {
    return (
      <div className="card">
        <div className="spinner"></div>
        <p className="text-center text-gray-600">Loading...</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Team Draft Evaluation</h2>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex justify-between text-sm text-gray-600 mb-2">
          <span>
            Flight {currentIndex + 1} of {interleavedFlights.length}
          </span>
          <span>{Math.round(progress)}% complete</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      {/* Flight Card */}
      <div className="bg-gradient-to-br from-blue-50 to-purple-50 p-8 rounded-xl shadow-lg mb-6">
        <div className="text-center mb-6">
          <h3 className="text-3xl font-bold text-gray-800 mb-2">
            {currentFlight.name || 'Flight'}
          </h3>
          <p className="text-xl text-gray-600">
            {currentFlight.origin} → {currentFlight.destination}
          </p>
        </div>

        <div className="grid grid-cols-2 gap-6 mb-6">
          <div className="bg-white p-4 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Departure</p>
            <p className="text-lg font-semibold">
              {formatTime(currentFlight.departure_time)}
            </p>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Arrival</p>
            <p className="text-lg font-semibold">
              {formatTime(currentFlight.arrival_time)}
            </p>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Duration</p>
            <p className="text-lg font-semibold">
              {formatDuration(currentFlight.duration_min)}
            </p>
          </div>
          <div className="bg-white p-4 rounded-lg">
            <p className="text-sm text-gray-600 mb-1">Stops</p>
            <p className="text-lg font-semibold">
              {currentFlight.stops}
            </p>
          </div>
        </div>

        <div className="text-center">
          <div className="text-4xl font-bold text-green-600 mb-2">
            ${currentFlight.price.toFixed(2)}
          </div>
          <p className="text-sm text-gray-600">Total Price</p>
        </div>
      </div>

      {/* Preference Buttons */}
      <div className="grid grid-cols-2 gap-4">
        <button
          onClick={() => handlePreference(false)}
          disabled={loading}
          className="bg-red-500 text-white px-8 py-4 rounded-lg hover:bg-red-600 transition-colors disabled:bg-gray-400 text-xl font-semibold"
        >
          👎 No
        </button>
        <button
          onClick={() => handlePreference(true)}
          disabled={loading}
          className="bg-green-500 text-white px-8 py-4 rounded-lg hover:bg-green-600 transition-colors disabled:bg-gray-400 text-xl font-semibold"
        >
          👍 Yes
        </button>
      </div>

      {error && (
        <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}
    </div>
  );
};

export default TeamDraft;
