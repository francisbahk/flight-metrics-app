/**
 * Traditional Flight Search Component
 * Google Flights-style form with structured inputs (for Manual method)
 */
import React, { useState } from 'react';
import { searchFlights } from '../api/flights';
import { format, addDays } from 'date-fns';

const TraditionalFlightSearch = ({ onSearchComplete }) => {
  const [formData, setFormData] = useState({
    origin: '',
    destination: '',
    departure_date: format(addDays(new Date(), 7), 'yyyy-MM-dd'),
    adults: 1,
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

      // Generate session ID
      const sessionId = `manual_${Date.now()}_${Math.random().toString(36).substring(7)}`;

      // Build natural language query for API (backend still uses Gemini for parsing)
      const query = `Find flights from ${formData.origin.toUpperCase()} to ${formData.destination.toUpperCase()} on ${formData.departure_date} for ${formData.adults} adult${formData.adults > 1 ? 's' : ''}`;

      // Call API
      const result = await searchFlights(query, sessionId);

      // Store session ID and method
      result.sessionId = sessionId;
      result.method = 'manual';

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

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">
        Search Flights
      </h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Origin */}
          <div>
            <label htmlFor="origin" className="block text-sm font-medium text-gray-700 mb-1">
              From (Origin)
            </label>
            <input
              type="text"
              id="origin"
              name="origin"
              value={formData.origin}
              onChange={handleInputChange}
              placeholder="JFK"
              maxLength={3}
              className="input-field w-full uppercase"
              required
            />
            <p className="text-xs text-gray-500 mt-1">3-letter airport code</p>
          </div>

          {/* Destination */}
          <div>
            <label htmlFor="destination" className="block text-sm font-medium text-gray-700 mb-1">
              To (Destination)
            </label>
            <input
              type="text"
              id="destination"
              name="destination"
              value={formData.destination}
              onChange={handleInputChange}
              placeholder="LAX"
              maxLength={3}
              className="input-field w-full uppercase"
              required
            />
            <p className="text-xs text-gray-500 mt-1">3-letter airport code</p>
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
              Passengers
            </label>
            <select
              id="adults"
              name="adults"
              value={formData.adults}
              onChange={handleInputChange}
              className="input-field w-full"
              required
            >
              {[1, 2, 3, 4, 5, 6, 7, 8, 9].map((num) => (
                <option key={num} value={num}>
                  {num} Adult{num > 1 ? 's' : ''}
                </option>
              ))}
            </select>
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
                Searching flights...
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

export default TraditionalFlightSearch;