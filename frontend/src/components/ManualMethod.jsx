/**
 * Manual Method Component
 * Traditional flight search with filters (baseline evaluation)
 */
import React, { useState, useEffect } from 'react';
import TraditionalFlightSearch from './TraditionalFlightSearch';
import FlightFilters from './FlightFilters';
import FlightCard from './FlightCard';
import Top5Panel from './Top5Panel';

const ManualMethod = ({ onComplete }) => {
  const [searchResults, setSearchResults] = useState(null);
  const [filters, setFilters] = useState(null);
  const [filteredFlights, setFilteredFlights] = useState([]);
  const [topFlights, setTopFlights] = useState([]);

  useEffect(() => {
    if (searchResults && searchResults.flights) {
      applyFilters(searchResults.flights, filters);
    }
  }, [searchResults, filters]);

  const applyFilters = (flights, activeFilters) => {
    if (!activeFilters) {
      setFilteredFlights(flights);
      return;
    }

    let filtered = [...flights];

    // Stops filter
    if (activeFilters.stops && activeFilters.stops.length > 0) {
      filtered = filtered.filter(flight => {
        const stops = flight.stops || 0;
        if (activeFilters.stops.includes('0') && stops === 0) return true;
        if (activeFilters.stops.includes('1') && stops === 1) return true;
        if (activeFilters.stops.includes('2+') && stops >= 2) return true;
        return false;
      });
    }

    // Price filter
    if (activeFilters.maxPrice) {
      filtered = filtered.filter(flight => flight.price <= activeFilters.maxPrice);
    }

    // Airlines filter
    if (activeFilters.airlines && activeFilters.airlines.length > 0) {
      filtered = filtered.filter(flight =>
        activeFilters.airlines.includes(flight.carrier)
      );
    }

    // Departure time filter
    if (activeFilters.departureTimeRange) {
      filtered = filtered.filter(flight => {
        const time = new Date(flight.departure_time).getHours();
        return time >= activeFilters.departureTimeRange[0] &&
               time <= activeFilters.departureTimeRange[1];
      });
    }

    // Arrival time filter
    if (activeFilters.arrivalTimeRange) {
      filtered = filtered.filter(flight => {
        const time = new Date(flight.arrival_time).getHours();
        return time >= activeFilters.arrivalTimeRange[0] &&
               time <= activeFilters.arrivalTimeRange[1];
      });
    }

    // Duration filter
    if (activeFilters.maxDuration) {
      filtered = filtered.filter(flight => {
        // Only filter flights that have valid duration data
        if (!flight.duration_minutes && flight.duration_minutes !== 0) {
          return true; // Keep flights with missing duration data
        }
        return flight.duration_minutes <= activeFilters.maxDuration;
      });
    }

    setFilteredFlights(filtered);
  };

  const handleSearchComplete = (results) => {
    setSearchResults(results);
    setTopFlights([]);
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
    console.log('Manual Method - Top 5 Rankings:', topFlights);

    // Call onComplete to move to next method
    if (onComplete) {
      onComplete({
        method: 'manual',
        rankings: topFlights,
        searchResults: searchResults,
      });
    }
  };

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="text-2xl font-bold mb-2 text-gray-800">
          Method 1: Manual Selection (Baseline)
        </h2>
        <p className="text-gray-600">
          Search for flights and use filters to find your preferred options. Select your top 5 flights and rank them.
        </p>
      </div>

      {!searchResults && (
        <TraditionalFlightSearch onSearchComplete={handleSearchComplete} />
      )}

      {searchResults && searchResults.flights && searchResults.flights.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Filters Sidebar */}
          <div className="lg:col-span-1">
            <FlightFilters
              flights={searchResults.flights}
              onFilterChange={setFilters}
            />
          </div>

          {/* Flights List */}
          <div className="lg:col-span-2 space-y-4">
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-gray-800">
                  Available Flights ({filteredFlights.length})
                </h3>
                <button
                  onClick={() => {
                    setSearchResults(null);
                    setTopFlights([]);
                  }}
                  className="btn-secondary text-sm"
                >
                  New Search
                </button>
              </div>

              <div className="space-y-3 max-h-[800px] overflow-y-auto">
                {filteredFlights.map((flight, index) => (
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

          {/* Top 5 Panel */}
          <div className="lg:col-span-1">
            <Top5Panel
              topFlights={topFlights}
              onReorder={handleReorder}
              onRemove={handleRemove}
              onSubmit={handleSubmit}
              methodName="Manual"
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default ManualMethod;
