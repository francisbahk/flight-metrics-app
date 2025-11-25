/**
 * LILO Method Component
 * 3-iteration interactive preference learning with natural language feedback
 *
 * Flow:
 * 1. User enters prompt
 * 2. Prompt 1: Generate initial preference questions → user answers
 * 3. Iteration 1: Random flights + Prompt 2 feedback → user gives feedback
 * 4. Iteration 2: Utility-ranked flights + Prompt 2 feedback → user gives feedback
 * 5. Iteration 3: Utility-ranked flights → user selects top 5
 */
import React, { useState } from 'react';
import FlightSearch from './FlightSearch';
import FlightCard from './FlightCard';
import Top5Panel from './Top5Panel';
import { generateLiloQuestions, rankWithFeedback } from '../api/flights';

const LILOMethod = ({ onComplete }) => {
  const [step, setStep] = useState('prompt'); // prompt, initial_questions, iteration1, iteration2, iteration3
  const [prompt, setPrompt] = useState('');
  const [searchResults, setSearchResults] = useState(null);
  const [initialQuestions, setInitialQuestions] = useState([]);
  const [initialAnswers, setInitialAnswers] = useState({});
  const [iteration, setIteration] = useState(0);
  const [currentFlights, setCurrentFlights] = useState([]);
  const [feedbackQuestions, setFeedbackQuestions] = useState([]);
  const [feedback, setFeedback] = useState('');
  const [allFeedback, setAllFeedback] = useState([]);
  const [topFlights, setTopFlights] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [sessionId] = useState(`lilo_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);

  // Prompt 1: Initial question generation using Gemini
  const generateInitialQuestions = async (userPrompt) => {
    setLoading(true);
    setError(null);
    try {
      const response = await generateLiloQuestions(userPrompt);
      setInitialQuestions(response.questions);
      setStep('initial_questions');
    } catch (err) {
      console.error('Error generating questions:', err);
      setError('Failed to generate questions. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSearchComplete = async (results) => {
    setSearchResults(results);
    const userPrompt = results.parsed_params?.user_prompt || prompt;
    setPrompt(userPrompt);

    // Generate initial questions
    await generateInitialQuestions(userPrompt);
  };

  const handleInitialQuestionsSubmit = async () => {
    // Start iteration 1 with random flights
    setLoading(true);
    try {
      // Shuffle flights for random selection
      const shuffled = [...searchResults.flights].sort(() => Math.random() - 0.5);
      const randomFlights = shuffled; // Show all flights

      setCurrentFlights(randomFlights);
      setIteration(1);
      setStep('iteration1');

      // Generate feedback questions (Prompt 2)
      await generateFeedbackQuestions(randomFlights);
    } catch (err) {
      setError('Failed to start iteration 1');
    } finally {
      setLoading(false);
    }
  };

  // Prompt 2: Generate feedback questions after user sees flights
  const generateFeedbackQuestions = async (flights) => {
    // TODO: Call backend API with Prompt 2
    // For now, use mock questions
    const mockQuestions = [
      "Which of these flights best matches your preferences?",
      "What aspects of the flights shown are most appealing to you?",
      "Are there any dealbreakers in the flights you see?",
      "How would you rank price vs. convenience for this trip?"
    ];

    setFeedbackQuestions(mockQuestions);
  };

  const handleFeedbackSubmit = async () => {
    if (!feedback.trim()) {
      setError('Please provide feedback');
      return;
    }

    setLoading(true);
    setError(null);
    const currentFeedback = feedback;
    setAllFeedback(prev => [...prev, currentFeedback]);

    try {
      // Call Gemini API to rank flights based on accumulated feedback
      const response = await rankWithFeedback(
        sessionId,
        searchResults.flights,
        currentFeedback,
        iteration === 1 ? initialAnswers : {}
      );

      const rankedFlights = response.ranked_flights || [];

      if (iteration === 1) {
        // Move to iteration 2 with Gemini-ranked flights
        setCurrentFlights(rankedFlights);
        setIteration(2);
        setStep('iteration2');
        setFeedback('');
        await generateFeedbackQuestions(rankedFlights);
      } else if (iteration === 2) {
        // Move to iteration 3 with final Gemini-ranked flights
        setCurrentFlights(rankedFlights);
        setIteration(3);
        setStep('iteration3');
        setFeedback('');
      }
    } catch (err) {
      console.error('Error processing feedback:', err);
      setError('Failed to process feedback. Please try again.');
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
    console.log('LILO Method - Top 5 Rankings:', topFlights);
    console.log('LILO Method - Prompt:', prompt);
    console.log('LILO Method - Initial Answers:', initialAnswers);
    console.log('LILO Method - All Feedback:', allFeedback);

    // Call onComplete to finish evaluation
    if (onComplete) {
      onComplete({
        method: 'lilo',
        rankings: topFlights,
        prompt: prompt,
        initialAnswers: initialAnswers,
        feedback: allFeedback,
        searchResults: searchResults,
      });
    }
  };

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="text-2xl font-bold mb-2 text-gray-800">
          Method 3: LILO (Interactive Preference Learning)
        </h2>
        <p className="text-gray-600">
          Answer preference questions and provide feedback over 3 iterations to help the system learn your preferences.
        </p>
        {iteration > 0 && (
          <div className="mt-3">
            <div className="flex gap-2">
              {[1, 2, 3].map(i => (
                <div
                  key={i}
                  className={`flex-1 h-2 rounded ${
                    iteration >= i ? 'bg-green-600' : 'bg-gray-300'
                  }`}
                />
              ))}
            </div>
            <p className="text-sm text-gray-600 mt-2">
              Iteration {iteration} of 3
            </p>
          </div>
        )}
      </div>

      {/* Step 1: Natural Language Prompt */}
      {step === 'prompt' && (
        <FlightSearch onSearchComplete={handleSearchComplete} />
      )}

      {/* Step 2: Initial Questions (Prompt 1) */}
      {step === 'initial_questions' && (
        <div className="card">
          <h3 className="text-xl font-bold mb-4 text-gray-800">
            Tell us about your preferences
          </h3>
          <p className="text-gray-600 mb-6">
            Answer these questions to help us understand what you're looking for:
          </p>

          <div className="space-y-4">
            {initialQuestions.map((question, idx) => (
              <div key={idx}>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {idx + 1}. {question}
                </label>
                <textarea
                  value={initialAnswers[idx] || ''}
                  onChange={(e) => setInitialAnswers(prev => ({
                    ...prev,
                    [idx]: e.target.value
                  }))}
                  className="input-field w-full h-20 resize-none"
                  placeholder="Your answer..."
                />
              </div>
            ))}
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mt-4">
              {error}
            </div>
          )}

          <button
            onClick={handleInitialQuestionsSubmit}
            disabled={loading || Object.keys(initialAnswers).length < initialQuestions.length}
            className="btn-primary w-full mt-6"
          >
            {loading ? 'Starting...' : 'Start LILO Iterations'}
          </button>
        </div>
      )}

      {/* Iterations 1 & 2: Flights + Feedback */}
      {(step === 'iteration1' || step === 'iteration2') && (
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-xl font-bold mb-2 text-gray-800">
              Iteration {iteration}: Review Flights & Provide Feedback
            </h3>
            <p className="text-gray-600">
              {iteration === 1
                ? 'Here are some flight options. Review them and answer the questions below.'
                : 'Based on your feedback, here are refined flight options.'}
            </p>
          </div>

          {/* Flights */}
          <div className="card">
            <h4 className="font-semibold text-gray-800 mb-4">
              Available Flights ({currentFlights.length})
            </h4>
            <div className="space-y-3 max-h-[500px] overflow-y-auto">
              {currentFlights.map((flight, index) => (
                <FlightCard
                  key={flight.id || index}
                  flight={flight}
                  isSelected={false}
                  onToggleSelect={() => {}}
                  showCheckbox={false}
                />
              ))}
            </div>
          </div>

          {/* Feedback Questions */}
          <div className="card">
            <h4 className="font-semibold text-gray-800 mb-4">
              Feedback Questions
            </h4>
            <div className="space-y-3 mb-4">
              {feedbackQuestions.map((question, idx) => (
                <p key={idx} className="text-sm text-gray-700">
                  • {question}
                </p>
              ))}
            </div>

            <label className="block text-sm font-medium text-gray-700 mb-2">
              Your Feedback:
            </label>
            <textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              className="input-field w-full h-32 resize-none"
              placeholder="Describe what you like or don't like about these options..."
            />

            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mt-4">
                {error}
              </div>
            )}

            <button
              onClick={handleFeedbackSubmit}
              disabled={loading}
              className="btn-primary w-full mt-4"
            >
              {loading ? 'Processing...' : `Continue to Iteration ${iteration + 1}`}
            </button>
          </div>
        </div>
      )}

      {/* Iteration 3: Final Selection */}
      {step === 'iteration3' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 space-y-4">
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-lg font-bold text-gray-800">
                    Final Recommendations ({currentFlights.length})
                  </h3>
                  <p className="text-sm text-gray-600 mt-1">
                    Select your top 5 flights
                  </p>
                </div>
              </div>

              <div className="space-y-3 max-h-[800px] overflow-y-auto">
                {currentFlights.map((flight, index) => (
                  <FlightCard
                    key={flight.id || index}
                    flight={flight}
                    isSelected={topFlights.some(f => f.id === flight.id)}
                    onToggleSelect={handleToggleSelect}
                  />
                ))}
              </div>
            </div>
          </div>

          <div className="lg:col-span-1">
            <Top5Panel
              topFlights={topFlights}
              onReorder={handleReorder}
              onRemove={handleRemove}
              onSubmit={handleSubmit}
              methodName="LILO"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default LILOMethod;
