/**
 * LILO Workflow Component
 * 2-round interactive preference learning workflow
 */
import React, { useState, useEffect } from 'react';
import { initLilo, submitLiloRound, getLiloFinal } from '../api/flights';

const LiloWorkflow = ({ flights, userPrompt, sessionId }) => {
  const [currentRound, setCurrentRound] = useState(0); // 0: init, 1: round 1, 2: round 2, 3: final
  const [liloSessionId, setLiloSessionId] = useState(null);
  const [roundFlights, setRoundFlights] = useState([]);
  const [selectedFlights, setSelectedFlights] = useState([]);
  const [feedback, setFeedback] = useState('');
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [finalRankings, setFinalRankings] = useState([]);
  const [feedbackSummary, setFeedbackSummary] = useState('');

  // Initialize LILO on mount
  useEffect(() => {
    initializeLilo();
  }, []);

  const initializeLilo = async () => {
    setLoading(true);
    setError(null);
    try {
      const liloId = `lilo_${sessionId || Date.now()}`;
      const response = await initLilo(liloId, flights, userPrompt);

      setLiloSessionId(response.session_id);
      setRoundFlights(response.flights_shown);
      setCurrentRound(1);
    } catch (err) {
      console.error('Failed to initialize LILO:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to initialize LILO');
    } finally {
      setLoading(false);
    }
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

  const handleSubmitRound = async () => {
    if (selectedFlights.length !== 5) {
      setError('Please select exactly 5 flights to rank');
      return;
    }

    if (!feedback.trim()) {
      setError('Please provide feedback about your preferences');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // User rankings are the indices in order of selection
      const userRankings = selectedFlights.map(f => f.originalIndex);

      const response = await submitLiloRound(
        liloSessionId,
        currentRound,
        userRankings,
        feedback
      );

      if (response.is_final_round) {
        // Get final rankings
        const finalResponse = await getLiloFinal(liloSessionId);
        setFinalRankings(finalResponse.final_rankings);
        setFeedbackSummary(finalResponse.feedback_summary);
        setCurrentRound(3);
      } else {
        // Move to next round
        setRoundFlights(response.flights_shown);
        setQuestions(response.questions || []);
        setSelectedFlights([]);
        setFeedback('');
        setCurrentRound(currentRound + 1);
      }
    } catch (err) {
      console.error('Failed to submit round:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to submit round');
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(price);
  };

  const formatDuration = (duration) => {
    if (!duration) return 'N/A';
    const hours = Math.floor(duration / 60);
    const minutes = duration % 60;
    return `${hours}h ${minutes}m`;
  };

  if (loading && currentRound === 0) {
    return (
      <div className="card text-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p className="text-gray-600">Initializing LILO...</p>
      </div>
    );
  }

  if (error && currentRound === 0) {
    return (
      <div className="card">
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      </div>
    );
  }

  // Round 1 or Round 2
  if (currentRound === 1 || currentRound === 2) {
    return (
      <div className="card">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">
          LILO - Round {currentRound} of 2
        </h2>
        <p className="text-gray-600 mb-6">
          Select your top 5 flights from the options below (in order of preference), then provide feedback about what you're looking for.
        </p>

        {/* Show questions from previous round */}
        {questions && questions.length > 0 && (
          <div className="bg-blue-50 border border-blue-200 p-4 rounded mb-6">
            <h3 className="font-semibold text-blue-900 mb-2">Questions to consider:</h3>
            <ul className="list-disc list-inside text-blue-800 space-y-1">
              {questions.map((q, idx) => (
                <li key={idx}>{q}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        {/* Flight selection */}
        <div className="mb-6">
          <h3 className="font-semibold text-gray-800 mb-3">
            Available Flights ({roundFlights.length}) - Select 5:
          </h3>
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {roundFlights.map((flight, index) => {
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
                      <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
                        <div>Departure: {flight.departure_time}</div>
                        <div>Arrival: {flight.arrival_time}</div>
                        <div>Duration: {formatDuration(flight.duration_minutes)}</div>
                        <div>Stops: {flight.stops}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-blue-600">
                        {formatPrice(flight.price)}
                      </div>
                      <div className="text-xs text-gray-500">{flight.cabin_class}</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Selected flights summary */}
        {selectedFlights.length > 0 && (
          <div className="mb-6 bg-gray-50 p-4 rounded">
            <h3 className="font-semibold text-gray-800 mb-2">
              Your Ranking ({selectedFlights.length}/5):
            </h3>
            <ol className="list-decimal list-inside text-sm text-gray-700 space-y-1">
              {selectedFlights.map((flight, idx) => (
                <li key={idx}>
                  {flight.origin} → {flight.destination} - {formatPrice(flight.price)}
                </li>
              ))}
            </ol>
          </div>
        )}

        {/* Feedback textarea */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Feedback (explain your preferences):
          </label>
          <textarea
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            placeholder="E.g., 'I prefer direct flights even if they cost more' or 'I want the cheapest option with at most 1 stop'"
            className="input-field w-full h-24 resize-none"
            required
          />
        </div>

        {/* Submit button */}
        <button
          onClick={handleSubmitRound}
          disabled={loading || selectedFlights.length !== 5}
          className="btn-primary w-full"
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing...
            </span>
          ) : (
            `Submit Round ${currentRound}`
          )}
        </button>
      </div>
    );
  }

  // Final results
  if (currentRound === 3) {
    return (
      <div className="card">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">
          LILO Results - Your Personalized Ranking
        </h2>

        {feedbackSummary && (
          <div className="bg-green-50 border border-green-200 p-4 rounded mb-6">
            <h3 className="font-semibold text-green-900 mb-2">Learned Preferences:</h3>
            <p className="text-green-800">{feedbackSummary}</p>
          </div>
        )}

        <p className="text-gray-600 mb-6">
          Based on your feedback across 2 rounds, here are your top 10 flights ranked by learned utility:
        </p>

        <div className="space-y-3">
          {finalRankings.map((flight, index) => (
            <div key={index} className="p-4 border-2 border-gray-200 rounded-lg bg-white">
              <div className="flex justify-between items-start">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="bg-green-600 text-white text-sm font-bold px-3 py-1 rounded">
                      #{index + 1}
                    </span>
                    <span className="font-semibold text-gray-800">
                      {flight.origin} → {flight.destination}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
                    <div>Departure: {flight.departure_time}</div>
                    <div>Arrival: {flight.arrival_time}</div>
                    <div>Duration: {formatDuration(flight.duration_minutes)}</div>
                    <div>Stops: {flight.stops}</div>
                    <div>Carrier: {flight.carrier}</div>
                    <div>Class: {flight.cabin_class}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-green-600">
                    {formatPrice(flight.price)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return null;
};

export default LiloWorkflow;
