/**
 * Flight Filters Component
 * Google Flights-style filtering (stops, price, airlines, time, duration)
 */
import React, { useState, useEffect } from 'react';

const FlightFilters = ({ flights, onFilterChange }) => {
  const [filters, setFilters] = useState({
    stops: [], // ['0', '1', '2+']
    maxPrice: null,
    airlines: [],
    departureTimeRange: [0, 24], // hours
    arrivalTimeRange: [0, 24], // hours
    maxDuration: null, // minutes
  });

  // Extract unique airlines from flights
  const uniqueAirlines = [...new Set(flights.map(f => f.carrier).filter(Boolean))].sort();

  // Get price range
  const prices = flights.map(f => f.price).filter(Boolean);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);

  // Get duration range
  const durations = flights.map(f => f.duration_minutes).filter(Boolean);
  const minDurationValue = durations.length > 0 ? Math.min(...durations) : 0;
  const maxDurationValue = durations.length > 0 ? Math.max(...durations) : 1000;

  // Initialize filters when flights change
  useEffect(() => {
    if (flights && flights.length > 0) {
      const prices = flights.map(f => f.price).filter(Boolean);
      const durations = flights.map(f => f.duration_minutes).filter(Boolean);

      if (prices.length > 0 && durations.length > 0) {
        const maxP = Math.max(...prices);
        const maxD = Math.max(...durations);

        setFilters(prev => ({
          ...prev,
          maxPrice: maxP,
          maxDuration: maxD,
        }));
      }
    }
  }, [flights]);

  useEffect(() => {
    // Apply filters whenever they change
    if (onFilterChange) {
      onFilterChange(filters);
    }
  }, [filters, onFilterChange]);

  const handleStopsChange = (stopValue) => {
    setFilters(prev => ({
      ...prev,
      stops: prev.stops.includes(stopValue)
        ? prev.stops.filter(s => s !== stopValue)
        : [...prev.stops, stopValue],
    }));
  };

  const handleAirlineChange = (airline) => {
    setFilters(prev => ({
      ...prev,
      airlines: prev.airlines.includes(airline)
        ? prev.airlines.filter(a => a !== airline)
        : [...prev.airlines, airline],
    }));
  };

  const resetFilters = () => {
    setFilters({
      stops: [],
      maxPrice: maxPrice,
      airlines: [],
      departureTimeRange: [0, 24],
      arrivalTimeRange: [0, 24],
      maxDuration: maxDurationValue,
    });
  };

  const activeFilterCount =
    filters.stops.length +
    (filters.maxPrice < maxPrice ? 1 : 0) +
    filters.airlines.length +
    (filters.departureTimeRange[0] > 0 || filters.departureTimeRange[1] < 24 ? 1 : 0) +
    (filters.arrivalTimeRange[0] > 0 || filters.arrivalTimeRange[1] < 24 ? 1 : 0) +
    (filters.maxDuration < maxDurationValue ? 1 : 0);

  return (
    <div className="bg-white rounded-lg shadow p-4 space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-bold text-gray-800">Filters</h3>
        {activeFilterCount > 0 && (
          <button
            onClick={resetFilters}
            className="text-sm text-blue-600 hover:text-blue-800"
          >
            Reset ({activeFilterCount})
          </button>
        )}
      </div>

      {/* Stops Filter */}
      <div>
        <h4 className="font-semibold text-gray-700 mb-2">Stops</h4>
        <div className="space-y-2">
          {['0', '1', '2+'].map((stop) => (
            <label key={stop} className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={filters.stops.includes(stop)}
                onChange={() => handleStopsChange(stop)}
                className="rounded"
              />
              <span className="text-sm text-gray-700">
                {stop === '0' ? 'Nonstop' : stop === '1' ? '1 stop' : '2+ stops'}
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Price Filter */}
      <div>
        <h4 className="font-semibold text-gray-700 mb-2">Max Price</h4>
        <input
          type="range"
          min={minPrice}
          max={maxPrice}
          value={filters.maxPrice || maxPrice}
          onChange={(e) => setFilters(prev => ({ ...prev, maxPrice: Number(e.target.value) }))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-gray-600 mt-1">
          <span>${minPrice}</span>
          <span className="font-semibold">${filters.maxPrice}</span>
          <span>${maxPrice}</span>
        </div>
      </div>

      {/* Airlines Filter */}
      <div>
        <h4 className="font-semibold text-gray-700 mb-2">Airlines</h4>
        <div className="max-h-40 overflow-y-auto space-y-2">
          {uniqueAirlines.map((airline) => (
            <label key={airline} className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={filters.airlines.includes(airline)}
                onChange={() => handleAirlineChange(airline)}
                className="rounded"
              />
              <span className="text-sm text-gray-700">{airline}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Departure Time Filter */}
      <div>
        <h4 className="font-semibold text-gray-700 mb-2">Departure Time</h4>
        <div className="flex gap-2 items-center">
          <input
            type="number"
            min="0"
            max="23"
            value={filters.departureTimeRange[0]}
            onChange={(e) => setFilters(prev => ({
              ...prev,
              departureTimeRange: [Number(e.target.value), prev.departureTimeRange[1]]
            }))}
            className="input-field w-20 text-sm"
          />
          <span className="text-gray-600">to</span>
          <input
            type="number"
            min="0"
            max="24"
            value={filters.departureTimeRange[1]}
            onChange={(e) => setFilters(prev => ({
              ...prev,
              departureTimeRange: [prev.departureTimeRange[0], Number(e.target.value)]
            }))}
            className="input-field w-20 text-sm"
          />
        </div>
        <p className="text-xs text-gray-500 mt-1">Hours (0-24)</p>
      </div>

      {/* Arrival Time Filter */}
      <div>
        <h4 className="font-semibold text-gray-700 mb-2">Arrival Time</h4>
        <div className="flex gap-2 items-center">
          <input
            type="number"
            min="0"
            max="23"
            value={filters.arrivalTimeRange[0]}
            onChange={(e) => setFilters(prev => ({
              ...prev,
              arrivalTimeRange: [Number(e.target.value), prev.arrivalTimeRange[1]]
            }))}
            className="input-field w-20 text-sm"
          />
          <span className="text-gray-600">to</span>
          <input
            type="number"
            min="0"
            max="24"
            value={filters.arrivalTimeRange[1]}
            onChange={(e) => setFilters(prev => ({
              ...prev,
              arrivalTimeRange: [prev.arrivalTimeRange[0], Number(e.target.value)]
            }))}
            className="input-field w-20 text-sm"
          />
        </div>
        <p className="text-xs text-gray-500 mt-1">Hours (0-24)</p>
      </div>

      {/* Duration Filter */}
      <div>
        <h4 className="font-semibold text-gray-700 mb-2">Max Duration</h4>
        <input
          type="range"
          min={minDurationValue}
          max={maxDurationValue}
          value={filters.maxDuration || maxDurationValue}
          onChange={(e) => setFilters(prev => ({ ...prev, maxDuration: Number(e.target.value) }))}
          className="w-full"
        />
        <div className="text-center text-sm font-semibold text-gray-700 mt-1">
          {filters.maxDuration && filters.maxDuration > 0
            ? `${Math.floor(filters.maxDuration / 60)}h ${filters.maxDuration % 60}m`
            : 'Any duration'}
        </div>
      </div>
    </div>
  );
};

export default FlightFilters;