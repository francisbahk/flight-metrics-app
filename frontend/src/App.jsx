/**
 * Main App Component
 * Orchestrates sequential evaluation workflow: Manual → LISTEN → LILO
 */
import React, { useState } from 'react';
import ManualMethod from './components/ManualMethod';
import LISTENMethod from './components/LISTENMethod';
import LILOMethod from './components/LILOMethod';

function App() {
  // Sequential evaluation state
  const [currentMethod, setCurrentMethod] = useState('manual'); // manual, listen, lilo, complete
  const [evaluationData, setEvaluationData] = useState({
    manual: null,
    listen: null,
    lilo: null,
  });

  // Handle Manual method completion
  const handleManualComplete = (data) => {
    console.log('Manual method completed:', data);
    setEvaluationData(prev => ({ ...prev, manual: data }));
    setCurrentMethod('listen');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Handle LISTEN method completion
  const handleLISTENComplete = (data) => {
    console.log('LISTEN method completed:', data);
    setEvaluationData(prev => ({ ...prev, listen: data }));
    setCurrentMethod('lilo');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Handle LILO method completion
  const handleLILOComplete = (data) => {
    console.log('LILO method completed:', data);
    setEvaluationData(prev => ({ ...prev, lilo: data }));
    setCurrentMethod('complete');
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // TODO: Submit all evaluation data to backend
    console.log('All evaluation data:', {
      manual: evaluationData.manual,
      listen: evaluationData.listen,
      lilo: data,
    });
  };

  // Reset evaluation (start over)
  const handleReset = () => {
    setCurrentMethod('manual');
    setEvaluationData({ manual: null, listen: null, lilo: null });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Get method info
  const getMethodInfo = () => {
    const methods = {
      manual: { number: 1, name: 'Manual (Baseline)', color: 'from-gray-600 to-gray-700' },
      listen: { number: 2, name: 'LISTEN (AI Ranking)', color: 'from-blue-600 to-indigo-600' },
      lilo: { number: 3, name: 'LILO (Interactive Learning)', color: 'from-green-600 to-teal-600' },
    };
    return methods[currentMethod] || { number: 0, name: '', color: '' };
  };

  const methodInfo = getMethodInfo();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-4xl font-bold text-gray-800">
            Flight Ranking Evaluation
          </h1>
          <p className="text-gray-600 mt-2">
            Sequential evaluation: Manual → LISTEN → LILO
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <div className="space-y-6">
          {/* Progress Indicator */}
          {currentMethod !== 'complete' && (
            <div className="card">
              <div className={`bg-gradient-to-r ${methodInfo.color} text-white px-6 py-4 rounded-lg`}>
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-2xl font-bold">
                      Method {methodInfo.number} of 3: {methodInfo.name}
                    </h2>
                    <p className="text-white/90 text-sm mt-1">
                      Complete all three methods to finish the evaluation
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold">{methodInfo.number}/3</div>
                  </div>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="mt-4">
                <div className="flex gap-2">
                  {[
                    { key: 'manual', label: 'Manual' },
                    { key: 'listen', label: 'LISTEN' },
                    { key: 'lilo', label: 'LILO' }
                  ].map((method, idx) => (
                    <div key={method.key} className="flex-1">
                      <div
                        className={`h-2 rounded-full ${
                          evaluationData[method.key]
                            ? 'bg-green-600'
                            : currentMethod === method.key
                            ? 'bg-blue-600'
                            : 'bg-gray-300'
                        }`}
                      />
                      <p className="text-xs text-gray-600 mt-1 text-center">
                        {method.label}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Manual Method */}
          {currentMethod === 'manual' && (
            <ManualMethod onComplete={handleManualComplete} />
          )}

          {/* LISTEN Method */}
          {currentMethod === 'listen' && (
            <LISTENMethod onComplete={handleLISTENComplete} />
          )}

          {/* LILO Method */}
          {currentMethod === 'lilo' && (
            <LILOMethod onComplete={handleLILOComplete} />
          )}

          {/* Completion Message */}
          {currentMethod === 'complete' && (
            <div className="card text-center py-12">
              <div className="text-6xl mb-6">🎉</div>
              <h2 className="text-3xl font-bold text-gray-800 mb-4">
                Evaluation Complete!
              </h2>
              <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
                Thank you for completing all three evaluation methods. Your responses have been recorded.
              </p>

              {/* Summary */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto mt-8">
                <div className="bg-gray-50 p-6 rounded-lg border-2 border-green-500">
                  <div className="text-3xl mb-3">✓</div>
                  <h3 className="font-semibold text-gray-800 mb-2">
                    Manual (Baseline)
                  </h3>
                  <p className="text-sm text-gray-600">
                    {evaluationData.manual?.rankings?.length || 0} flights ranked
                  </p>
                </div>
                <div className="bg-blue-50 p-6 rounded-lg border-2 border-green-500">
                  <div className="text-3xl mb-3">✓</div>
                  <h3 className="font-semibold text-gray-800 mb-2">
                    LISTEN (AI Ranking)
                  </h3>
                  <p className="text-sm text-gray-600">
                    {evaluationData.listen?.rankings?.length || 0} flights ranked
                  </p>
                </div>
                <div className="bg-green-50 p-6 rounded-lg border-2 border-green-500">
                  <div className="text-3xl mb-3">✓</div>
                  <h3 className="font-semibold text-gray-800 mb-2">
                    LILO (Interactive)
                  </h3>
                  <p className="text-sm text-gray-600">
                    {evaluationData.lilo?.rankings?.length || 0} flights ranked
                  </p>
                </div>
              </div>

              <button
                onClick={handleReset}
                className="btn-primary mt-8"
              >
                Start New Evaluation
              </button>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white mt-16 border-t border-gray-200">
        <div className="container mx-auto px-4 py-6 text-center text-gray-600">
          <p className="text-sm">
            Flight Ranking Evaluation v2.0 - Built with FastAPI, React, and Amadeus API
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
