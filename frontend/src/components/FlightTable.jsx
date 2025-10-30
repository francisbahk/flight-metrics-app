/**
 * FlightTable Component
 * Displays flight search results in a sortable table
 */
import React, { useState } from 'react';
import { format } from 'date-fns';

const FlightTable = ({ flights, onSelectFlight, selectedFlights = [] }) => {
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

  // Sorting function - must be called before any early returns
  const sortedFlights = React.useMemo(() => {
    if (!flights || flights.length === 0) {
      return [];
    }
    let sortableFlights = [...flights];
    if (sortConfig.key !== null) {
      sortableFlights.sort((a, b) => {
        let aVal = a[sortConfig.key];
        let bVal = b[sortConfig.key];

        // Handle nested properties or special cases
        if (sortConfig.key === 'departure_time' || sortConfig.key === 'arrival_time') {
          aVal = new Date(aVal).getTime();
          bVal = new Date(bVal).getTime();
        }

        if (aVal < bVal) {
          return sortConfig.direction === 'asc' ? -1 : 1;
        }
        if (aVal > bVal) {
          return sortConfig.direction === 'asc' ? 1 : -1;
        }
        return 0;
      });
    }
    return sortableFlights;
  }, [flights, sortConfig]);

  // Early return if no flights
  if (!flights || flights.length === 0) {
    return (
      <div className="card text-center py-8">
        <p className="text-gray-500">No flights to display. Search for flights to get started.</p>
      </div>
    );
  }

  // Request sort
  const requestSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  // Get sort indicator
  const getSortIndicator = (columnKey) => {
    if (sortConfig.key !== columnKey) {
      return '↕';
    }
    return sortConfig.direction === 'asc' ? '↑' : '↓';
  };

  // Format time for display
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

  // Handle flight selection
  const handleFlightSelect = (flight) => {
    if (onSelectFlight) {
      onSelectFlight(flight);
    }
  };

  // Check if flight is selected
  const isSelected = (flightId) => {
    return selectedFlights.some((f) => f.id === flightId);
  };

  // Calculate summary statistics
  const avgPrice = (flights.reduce((sum, f) => sum + f.price, 0) / flights.length).toFixed(2);
  const minPrice = Math.min(...flights.map((f) => f.price)).toFixed(2);
  const maxPrice = Math.max(...flights.map((f) => f.price)).toFixed(2);

  return (
    <div className="card">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold text-gray-800">
          Flight Results ({flights.length})
        </h2>
        {selectedFlights.length > 0 && (
          <span className="text-sm text-blue-600 font-medium">
            {selectedFlights.length} selected
          </span>
        )}
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full">
          <thead>
            <tr>
              {onSelectFlight && (
                <th className="table-header w-12">
                  Select
                </th>
              )}
              <th
                className="table-header cursor-pointer hover:bg-gray-200"
                onClick={() => requestSort('name')}
              >
                Airline {getSortIndicator('name')}
              </th>
              <th
                className="table-header cursor-pointer hover:bg-gray-200"
                onClick={() => requestSort('origin')}
              >
                Route {getSortIndicator('origin')}
              </th>
              <th
                className="table-header cursor-pointer hover:bg-gray-200"
                onClick={() => requestSort('departure_time')}
              >
                Departure {getSortIndicator('departure_time')}
              </th>
              <th
                className="table-header cursor-pointer hover:bg-gray-200"
                onClick={() => requestSort('arrival_time')}
              >
                Arrival {getSortIndicator('arrival_time')}
              </th>
              <th
                className="table-header cursor-pointer hover:bg-gray-200"
                onClick={() => requestSort('duration_min')}
              >
                Duration {getSortIndicator('duration_min')}
              </th>
              <th
                className="table-header cursor-pointer hover:bg-gray-200"
                onClick={() => requestSort('stops')}
              >
                Stops {getSortIndicator('stops')}
              </th>
              <th
                className="table-header cursor-pointer hover:bg-gray-200"
                onClick={() => requestSort('price')}
              >
                Price {getSortIndicator('price')}
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedFlights.map((flight) => (
              <tr
                key={flight.id}
                className={`hover:bg-gray-50 ${isSelected(flight.id) ? 'bg-blue-50' : ''}`}
              >
                {onSelectFlight && (
                  <td className="table-cell text-center">
                    <input
                      type="checkbox"
                      checked={isSelected(flight.id)}
                      onChange={() => handleFlightSelect(flight)}
                      className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                    />
                  </td>
                )}
                <td className="table-cell font-medium">
                  {flight.name || 'N/A'}
                </td>
                <td className="table-cell">
                  {flight.origin} → {flight.destination}
                </td>
                <td className="table-cell">
                  {formatTime(flight.departure_time)}
                </td>
                <td className="table-cell">
                  {formatTime(flight.arrival_time)}
                </td>
                <td className="table-cell">
                  {formatDuration(flight.duration_min)}
                </td>
                <td className="table-cell text-center">
                  {flight.stops}
                </td>
                <td className="table-cell font-semibold text-green-600">
                  ${flight.price.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary Statistics */}
      <div className="mt-6 pt-4 border-t border-gray-200">
        <h3 className="text-lg font-semibold text-gray-700 mb-3">Summary Statistics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Average Price</p>
            <p className="text-2xl font-bold text-blue-600">${avgPrice}</p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Minimum Price</p>
            <p className="text-2xl font-bold text-green-600">${minPrice}</p>
          </div>
          <div className="bg-orange-50 p-4 rounded-lg">
            <p className="text-sm text-gray-600">Maximum Price</p>
            <p className="text-2xl font-bold text-orange-600">${maxPrice}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FlightTable;
