/**
 * FlightSearch Component
 * Form for searching flights via Amadeus API
 */
import React, { useState } from 'react';
import { searchFlights } from '../api/flights';
import { format, addDays } from 'date-fns';

const FlightSearch = ({ onSearchComplete }) => {
  const [formData, setFormData] = useState({
    origin: '',
    destination: '',
    departure_date: format(addDays(new Date(), 7), 'yyyy-MM-dd'), // Default to 1 week from now
    adults: 1,
    max_results: 10,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      // Validate inputs
      if (!formData.origin || !formData.destination) {
        throw new Error('Please enter both origin and destination');
      }

      if (formData.origin.length !== 3 || formData.destination.length !== 3) {
        throw new Error('Airport codes must be 3 letters (e.g., JFK, LAX)');
      }

      // Call API
      const result = await searchFlights(formData);

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

  // Quick example buttons
  const loadExample = (origin, destination) => {
    setFormData((prev) => ({
      ...prev,
      origin,
      destination,
      departure_date: format(addDays(new Date(), 7), 'yyyy-MM-dd'),
    }));
  };

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Search Flights</h2>

      {/* Quick Examples */}
      <div className="mb-6">
        <p className="text-sm text-gray-600 mb-2">Quick examples:</p>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => loadExample('JFK', 'LAX')}
            className="btn-secondary text-sm"
          >
            JFK → LAX
          </button>
          <button
            type="button"
            onClick={() => loadExample('SFO', 'JFK')}
            className="btn-secondary text-sm"
          >
            SFO → JFK
          </button>
          <button
            type="button"
            onClick={() => loadExample('ORD', 'MIA')}
            className="btn-secondary text-sm"
          >
            ORD → MIA
          </button>
          <button
            type="button"
            onClick={() => loadExample('LAX', 'LHR')}
            className="btn-secondary text-sm"
          >
            LAX → LHR
          </button>
        </div>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Origin */}
          <div>
            <label htmlFor="origin" className="block text-sm font-medium text-gray-700 mb-1">
              Origin Airport Code
            </label>
            <input
              type="text"
              id="origin"
              name="origin"
              value={formData.origin}
              onChange={handleInputChange}
              placeholder="e.g., JFK"
              maxLength={3}
              className="input-field w-full uppercase"
              required
            />
          </div>

          {/* Destination */}
          <div>
            <label htmlFor="destination" className="block text-sm font-medium text-gray-700 mb-1">
              Destination Airport Code
            </label>
            <input
              type="text"
              id="destination"
              name="destination"
              value={formData.destination}
              onChange={handleInputChange}
              placeholder="e.g., LAX"
              maxLength={3}
              className="input-field w-full uppercase"
              required
            />
          </div>

          {/* Departure Date */}
          <div>
            <label htmlFor="departure_date" className="block text-sm font-medium text-gray-700 mb-1">
              Departure Date
            </label>
            <input
              type="date"
              id="departure_date"
              name="departure_date"
              value={formData.departure_date}
              onChange={handleInputChange}
              min={format(new Date(), 'yyyy-MM-dd')}
              className="input-field w-full"
              required
            />
          </div>

          {/* Adults */}
          <div>
            <label htmlFor="adults" className="block text-sm font-medium text-gray-700 mb-1">
              Number of Adults
            </label>
            <input
              type="number"
              id="adults"
              name="adults"
              value={formData.adults}
              onChange={handleInputChange}
              min={1}
              max={9}
              className="input-field w-full"
              required
            />
          </div>

          {/* Max Results */}
          <div>
            <label htmlFor="max_results" className="block text-sm font-medium text-gray-700 mb-1">
              Maximum Results
            </label>
            <input
              type="number"
              id="max_results"
              name="max_results"
              value={formData.max_results}
              onChange={handleInputChange}
              min={1}
              max={50}
              className="input-field w-full"
              required
            />
          </div>
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
                Searching...
              </span>
            ) : (
              'Search Flights'
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default FlightSearch;
