/**
 * Evaluation Workflow Component
 * Person A vs Person B vs Algorithms comparison
 */
import React, { useState } from 'react';
import {
  startEvaluationSession,
  submitPersonARankings,
  submitPersonBRankings,
  compareRankings,
  rankFlightsListenU,
  initLilo,
  submitLiloRound,
  getLiloFinal,
} from '../api/flights';

const EvaluationWorkflow = ({ flights, onComplete }) => {
  const [step, setStep] = useState('select_mode'); // select_mode, person_a, person_b, results
  const [mode, setMode] = useState(null); // 'person_a' or 'person_b'
  const [evalSessionId, setEvalSessionId] = useState(null);
  const [userId, setUserId] = useState('');
  const [prompt, setPrompt] = useState('');
  const [selectedFlights, setSelectedFlights] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [comparisonResults, setComparisonResults] = useState(null);

  const handleStartPersonA = () => {
    setMode('person_a');
    setStep('person_a');
  };

  const handleStartPersonB = () => {
    setMode('person_b');
    setStep('person_b');
  };

  const handleFlightSelect = (flight, index) => {
    const flightWithIndex = { ...flight, originalIndex: index };

    setSelectedFlights((prev) => {
      const isSelected = prev.some((f) => f.originalIndex === index);
      if (isSelected) {
        return prev.filter((f) => f.originalIndex !== index);
      } else if (prev.length < 5) {
        return [...prev, flightWithIndex];
      }
      return prev;
    });
  };

  const handleSubmitPersonA = async () => {
    if (!userId.trim()) {
      setError('Please enter your user ID');
      return;
    }

    if (!prompt.trim()) {
      setError('Please enter your flight preferences');
      return;
    }

    if (selectedFlights.length !== 5) {
      setError('Please select exactly 5 flights');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const sessionId = `eval_${Date.now()}_${Math.random().toString(36).substring(7)}`;

      // Start evaluation session
      await startEvaluationSession(sessionId, userId, prompt);

      // Submit Person A rankings
      await submitPersonARankings(sessionId, selectedFlights);

      setEvalSessionId(sessionId);
      setStep('results');

      // Run algorithms in background and compare
      await runAlgorithmsAndCompare(sessionId);
    } catch (err) {
      console.error('Failed to submit Person A rankings:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to submit rankings');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmitPersonB = async () => {
    if (!userId.trim()) {
      setError('Please enter your user ID');
      return;
    }

    if (!evalSessionId.trim()) {
      setError('Please enter the evaluation session ID from Person A');
      return;
    }

    if (selectedFlights.length !== 5) {
      setError('Please select exactly 5 flights');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Submit Person B rankings
      await submitPersonBRankings(evalSessionId, userId, selectedFlights);

      setStep('results');

      // Compare rankings
      const results = await compareRankings(evalSessionId);
      setComparisonResults(results);
    } catch (err) {
      console.error('Failed to submit Person B rankings:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to submit rankings');
    } finally {
      setLoading(false);
    }
  };

  const runAlgorithmsAndCompare = async (sessionId) => {
    try {
      // Note: In production, these would run in parallel or be triggered by backend
      // For now, we'll just show that Person A submitted successfully

      // The backend would automatically run LISTEN-U and LILO
      // and store the results in the evaluation session

      // For this demo, we'll just compare what we have
      const results = await compareRankings(sessionId);
      setComparisonResults(results);
    } catch (err) {
      console.error('Failed to run algorithms:', err);
    }
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(price);
  };

  // Select mode
  if (step === 'select_mode') {
    return (
      <div className="card">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">
          Evaluation Experiment
        </h2>
        <p className="text-gray-600 mb-6">
          Compare human preferences (Person A) with another person's guess (Person B) and algorithm rankings (LISTEN-U, LILO).
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <button
            onClick={handleStartPersonA}
            className="bg-blue-600 text-white p-6 rounded-lg hover:bg-blue-700 transition-colors text-left"
          >
            <h3 className="text-xl font-semibold mb-2">
              I am Person A
            </h3>
            <p className="text-blue-100 text-sm">
              Enter your preferences and rank your top 5 flights. Share the session ID with Person B.
            </p>
          </button>

          <button
            onClick={handleStartPersonB}
            className="bg-purple-600 text-white p-6 rounded-lg hover:bg-purple-700 transition-colors text-left"
          >
            <h3 className="text-xl font-semibold mb-2">
              I am Person B
            </h3>
            <p className="text-purple-100 text-sm">
              Given Person A's preferences, guess their top 5 flight choices.
            </p>
          </button>
        </div>
      </div>
    );
  }

  // Person A workflow
  if (step === 'person_a') {
    return (
      <div className="card">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">
          Person A - Ground Truth Preferences
        </h2>
        <p className="text-gray-600 mb-6">
          Enter your flight preferences and select your top 5 flights.
        </p>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        <div className="space-y-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Your User ID:
            </label>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="e.g., alice123"
              className="input-field w-full"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Your Flight Preferences:
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="E.g., 'I want a direct flight in the morning, prefer economy class, budget is flexible'"
              className="input-field w-full h-24 resize-none"
              required
            />
          </div>
        </div>

        <h3 className="font-semibold text-gray-800 mb-3">
          Select Your Top 5 Flights ({selectedFlights.length}/5):
        </h3>

        <div className="space-y-3 max-h-96 overflow-y-auto mb-6">
          {flights.map((flight, index) => {
            const isSelected = selectedFlights.some(f => f.originalIndex === index);
            const selectionOrder = selectedFlights.findIndex(f => f.originalIndex === index);

            return (
              <div
                key={index}
                onClick={() => handleFlightSelect(flight, index)}
                className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      {isSelected && (
                        <span className="bg-blue-600 text-white text-xs font-bold px-2 py-1 rounded">
                          #{selectionOrder + 1}
                        </span>
                      )}
                      <span className="font-semibold text-gray-800">
                        {flight.origin} → {flight.destination}
                      </span>
                    </div>
                    <div className="text-sm text-gray-600">
                      {flight.departure_time} - {flight.arrival_time} | {flight.stops} stops
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xl font-bold text-blue-600">
                      {formatPrice(flight.price)}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <button
          onClick={handleSubmitPersonA}
          disabled={loading || selectedFlights.length !== 5}
          className="btn-primary w-full"
        >
          {loading ? 'Submitting...' : 'Submit Preferences'}
        </button>
      </div>
    );
  }

  // Person B workflow
  if (step === 'person_b') {
    return (
      <div className="card">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">
          Person B - Guess Person A's Preferences
        </h2>
        <p className="text-gray-600 mb-6">
          Based on Person A's stated preferences, select the 5 flights you think they would choose.
        </p>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        <div className="space-y-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Your User ID:
            </label>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="e.g., bob456"
              className="input-field w-full"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Person A's Session ID:
            </label>
            <input
              type="text"
              value={evalSessionId}
              onChange={(e) => setEvalSessionId(e.target.value)}
              placeholder="eval_xxxxx"
              className="input-field w-full"
              required
            />
          </div>
        </div>

        <h3 className="font-semibold text-gray-800 mb-3">
          Select 5 Flights You Think Person A Would Choose ({selectedFlights.length}/5):
        </h3>

        <div className="space-y-3 max-h-96 overflow-y-auto mb-6">
          {flights.map((flight, index) => {
            const isSelected = selectedFlights.some(f => f.originalIndex === index);
            const selectionOrder = selectedFlights.findIndex(f => f.originalIndex === index);

            return (
              <div
                key={index}
                onClick={() => handleFlightSelect(flight, index)}
                className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                  isSelected
                    ? 'border-purple-500 bg-purple-50'
                    : 'border-gray-200 hover:border-purple-300 hover:bg-gray-50'
                }`}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      {isSelected && (
                        <span className="bg-purple-600 text-white text-xs font-bold px-2 py-1 rounded">
                          #{selectionOrder + 1}
                        </span>
                      )}
                      <span className="font-semibold text-gray-800">
                        {flight.origin} → {flight.destination}
                      </span>
                    </div>
                    <div className="text-sm text-gray-600">
                      {flight.departure_time} - {flight.arrival_time} | {flight.stops} stops
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xl font-bold text-purple-600">
                      {formatPrice(flight.price)}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        <button
          onClick={handleSubmitPersonB}
          disabled={loading || selectedFlights.length !== 5}
          className="btn-primary w-full bg-purple-600 hover:bg-purple-700"
        >
          {loading ? 'Submitting...' : 'Submit Guesses'}
        </button>
      </div>
    );
  }

  // Results
  if (step === 'results') {
    return (
      <div className="card">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">
          Evaluation Results
        </h2>

        {mode === 'person_a' && (
          <div className="bg-green-50 border border-green-200 p-4 rounded mb-6">
            <h3 className="font-semibold text-green-900 mb-2">Success!</h3>
            <p className="text-green-800 mb-2">
              Your preferences have been recorded.
            </p>
            <p className="text-green-800">
              Share this session ID with Person B: <br />
              <code className="bg-white px-2 py-1 rounded font-mono text-sm">
                {evalSessionId}
              </code>
            </p>
          </div>
        )}

        {comparisonResults && (
          <div className="space-y-4">
            <h3 className="font-semibold text-gray-800">Comparison Metrics:</h3>
            <pre className="bg-gray-100 p-4 rounded overflow-auto text-sm">
              {JSON.stringify(comparisonResults, null, 2)}
            </pre>
          </div>
        )}

        {!comparisonResults && mode === 'person_a' && (
          <p className="text-gray-600">
            Algorithms are running in the background. Person B can now submit their guesses using your session ID.
          </p>
        )}
      </div>
    );
  }

  return null;
};

export default EvaluationWorkflow;
