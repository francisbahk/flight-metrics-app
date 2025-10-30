/**
 * Main App Component
 * Orchestrates flight search and evaluation workflows
 */
import React, { useState } from 'react';
import FlightSearch from './components/FlightSearch';
import FlightTable from './components/FlightTable';
import ListenRanking from './components/ListenRanking';
import TeamDraft from './components/TeamDraft';
import ListenAlgorithms from './components/ListenAlgorithms';

function App() {
  // State management
  const [searchResults, setSearchResults] = useState(null);
  const [selectedFlights, setSelectedFlights] = useState([]);
  const [evaluationMode, setEvaluationMode] = useState(null); // null, 'listen', 'teamdraft', or 'listen-algorithms'

  // Handle search completion
  const handleSearchComplete = (results) => {
    setSearchResults(results);
    setSelectedFlights([]);
    setEvaluationMode(null);
  };

  // Handle flight selection (toggle)
  const handleFlightSelect = (flight) => {
    setSelectedFlights((prev) => {
      const isSelected = prev.some((f) => f.id === flight.id);
      if (isSelected) {
        return prev.filter((f) => f.id !== flight.id);
      } else {
        return [...prev, flight];
      }
    });
  };

  // Start evaluation mode
  const startEvaluation = (mode) => {
    if (selectedFlights.length === 0) {
      alert('Please select at least one flight to evaluate');
      return;
    }
    setEvaluationMode(mode);
  };

  // Handle Team Draft completion
  const handleTeamDraftComplete = (results) => {
    console.log('Team Draft completed:', results);
    // Could add additional logic here, like showing a success message
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-4xl font-bold text-gray-800">
            ✈️ Flight Metrics
          </h1>
          <p className="text-gray-600 mt-2">
            Search flights and evaluate ranking algorithms
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="space-y-8">
          {/* Search Section */}
          <section>
            <FlightSearch onSearchComplete={handleSearchComplete} />
          </section>

          {/* Results Section */}
          {searchResults && searchResults.flights && searchResults.flights.length > 0 && (
            <>
              <section>
                <FlightTable
                  flights={searchResults.flights}
                  onSelectFlight={handleFlightSelect}
                  selectedFlights={selectedFlights}
                />
              </section>

              {/* Evaluation Mode Selection */}
              {!evaluationMode && selectedFlights.length > 0 && (
                <section className="card">
                  <h2 className="text-2xl font-bold mb-4 text-gray-800">
                    Evaluate Selected Flights
                  </h2>
                  <p className="text-gray-600 mb-6">
                    Choose an evaluation method to rank or compare the selected flights.
                  </p>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <button
                      onClick={() => startEvaluation('listen-algorithms')}
                      className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-6 rounded-lg hover:from-blue-700 hover:to-indigo-700 transition-colors text-left border-2 border-yellow-300"
                    >
                      <h3 className="text-xl font-semibold mb-2">
                        🤖 LISTEN-U & LISTEN-T
                      </h3>
                      <p className="text-blue-100 text-sm">
                        Automated ranking using AI algorithms (Utility Refinement & Tournament)
                      </p>
                      <span className="inline-block mt-2 bg-yellow-400 text-blue-900 text-xs font-bold px-2 py-1 rounded">
                        NEW!
                      </span>
                    </button>
                    <button
                      onClick={() => startEvaluation('listen')}
                      className="bg-blue-600 text-white p-6 rounded-lg hover:bg-blue-700 transition-colors text-left"
                    >
                      <h3 className="text-xl font-semibold mb-2">
                        🎯 LISTEN Ranking
                      </h3>
                      <p className="text-blue-100 text-sm">
                        Rank flights by dragging and dropping them in order of preference
                      </p>
                    </button>
                    <button
                      onClick={() => startEvaluation('teamdraft')}
                      className="bg-purple-600 text-white p-6 rounded-lg hover:bg-purple-700 transition-colors text-left"
                    >
                      <h3 className="text-xl font-semibold mb-2">
                        ⚔️ Team Draft
                      </h3>
                      <p className="text-purple-100 text-sm">
                        Compare two ranking algorithms through interleaved evaluation
                      </p>
                    </button>
                  </div>
                </section>
              )}

              {/* LISTEN Ranking Section */}
              {evaluationMode === 'listen' && (
                <section>
                  <div className="mb-4">
                    <button
                      onClick={() => setEvaluationMode(null)}
                      className="btn-secondary"
                    >
                      ← Back to Results
                    </button>
                  </div>
                  <ListenRanking flights={selectedFlights} />
                </section>
              )}

              {/* Team Draft Section */}
              {evaluationMode === 'teamdraft' && (
                <section>
                  <div className="mb-4">
                    <button
                      onClick={() => setEvaluationMode(null)}
                      className="btn-secondary"
                    >
                      ← Back to Results
                    </button>
                  </div>
                  <TeamDraft
                    selectedFlights={selectedFlights}
                    onComplete={handleTeamDraftComplete}
                  />
                </section>
              )}

              {/* LISTEN Algorithms Section */}
              {evaluationMode === 'listen-algorithms' && (
                <section>
                  <div className="mb-4">
                    <button
                      onClick={() => setEvaluationMode(null)}
                      className="btn-secondary"
                    >
                      ← Back to Results
                    </button>
                  </div>
                  <ListenAlgorithms flights={selectedFlights} />
                </section>
              )}
            </>
          )}

          {/* No Results Message */}
          {searchResults && searchResults.flights && searchResults.flights.length === 0 && (
            <section className="card text-center py-8">
              <p className="text-gray-600">
                No flights found. Try adjusting your search criteria.
              </p>
            </section>
          )}

          {/* Welcome Message */}
          {!searchResults && (
            <section className="card text-center py-12">
              <h2 className="text-3xl font-bold text-gray-800 mb-4">
                Welcome to Flight Metrics
              </h2>
              <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
                Search for flights using the Amadeus API and evaluate different ranking
                algorithms using LISTEN or Team Draft evaluation methods.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto mt-8">
                <div className="bg-blue-50 p-6 rounded-lg">
                  <div className="text-4xl mb-3">🔍</div>
                  <h3 className="font-semibold text-gray-800 mb-2">
                    Search Flights
                  </h3>
                  <p className="text-sm text-gray-600">
                    Find real flight options using the Amadeus Flight API
                  </p>
                </div>
                <div className="bg-purple-50 p-6 rounded-lg">
                  <div className="text-4xl mb-3">📊</div>
                  <h3 className="font-semibold text-gray-800 mb-2">
                    Rank & Evaluate
                  </h3>
                  <p className="text-sm text-gray-600">
                    Use LISTEN or Team Draft to evaluate flight rankings
                  </p>
                </div>
                <div className="bg-green-50 p-6 rounded-lg">
                  <div className="text-4xl mb-3">💾</div>
                  <h3 className="font-semibold text-gray-800 mb-2">
                    Store Results
                  </h3>
                  <p className="text-sm text-gray-600">
                    All evaluations are saved to the database for analysis
                  </p>
                </div>
              </div>
            </section>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white mt-16 border-t border-gray-200">
        <div className="container mx-auto px-4 py-6 text-center text-gray-600">
          <p className="text-sm">
            Flight Metrics v1.0 - Built with FastAPI, React, and Amadeus API
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
