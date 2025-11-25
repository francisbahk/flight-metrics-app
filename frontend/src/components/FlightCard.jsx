/**
 * Enhanced Flight Card Component
 * Shows ALL Amadeus metrics with airline prominent and expandable details
 */
import React, { useState } from 'react';

const FlightCard = ({ flight, isSelected, onToggleSelect, showCheckbox = true }) => {
  const [showDetails, setShowDetails] = useState(false);

  const formatPrice = (price) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(price);
  };

  const formatDuration = (minutes) => {
    if (!minutes) return 'N/A';
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
  };

  const formatTime = (timeString) => {
    if (!timeString) return 'N/A';
    try {
      const date = new Date(timeString);
      return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return timeString;
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    try {
      const date = new Date(dateString);
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      });
    } catch {
      return dateString;
    }
  };

  const copyFlightInfo = () => {
    const info = `${flight.carrier} ${flight.flight_number} - ${flight.origin} to ${flight.destination} - ${formatPrice(flight.price)} - ${formatDuration(flight.duration_minutes)} - ${flight.stops} stops`;
    navigator.clipboard.writeText(info);
    // Optional: Show a brief confirmation (you could add a toast notification here)
  };

  return (
    <div
      className={`border-2 rounded-lg p-4 transition-all ${
        isSelected
          ? 'border-blue-500 bg-blue-50'
          : 'border-gray-200 bg-white hover:border-gray-300'
      }`}
    >
      {/* Main Flight Info */}
      <div className="flex items-start justify-between gap-4">
        {/* Checkbox - only show when showCheckbox is true */}
        {showCheckbox && (
          <div className="pt-1">
            <input
              type="checkbox"
              checked={isSelected}
              onChange={() => onToggleSelect(flight)}
              className="w-5 h-5 rounded cursor-pointer"
            />
          </div>
        )}

        {/* Flight Details */}
        <div className={`flex-1 grid grid-cols-1 md:grid-cols-12 gap-4 ${!showCheckbox ? 'ml-0' : ''}`}>
          {/* Airline & Flight Number */}
          <div className="md:col-span-3">
            <div className="font-bold text-lg text-gray-900">
              {flight.carrier || 'Unknown Airline'}
            </div>
            <div className="text-sm text-gray-600">
              {flight.flight_number || 'N/A'}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {flight.cabin_class || 'Economy'}
            </div>
          </div>

          {/* Route & Times */}
          <div className="md:col-span-5">
            <div className="flex items-center gap-4">
              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {formatTime(flight.departure_time)}
                </div>
                <div className="text-sm text-gray-600">{flight.origin}</div>
                <div className="text-xs text-gray-500">
                  {formatDate(flight.departure_time)}
                </div>
              </div>

              <div className="flex-1">
                <div className="text-center text-sm text-gray-600">
                  {formatDuration(flight.duration_minutes)}
                </div>
                <div className="h-px bg-gray-300 my-1 relative">
                  <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white px-2 text-xs text-gray-500">
                    {flight.stops === 0 ? 'Nonstop' : `${flight.stops} stop${flight.stops > 1 ? 's' : ''}`}
                  </div>
                </div>
              </div>

              <div>
                <div className="text-2xl font-bold text-gray-900">
                  {formatTime(flight.arrival_time)}
                </div>
                <div className="text-sm text-gray-600">{flight.destination}</div>
                <div className="text-xs text-gray-500">
                  {formatDate(flight.arrival_time)}
                </div>
              </div>
            </div>
          </div>

          {/* Price */}
          <div className="md:col-span-2 text-right">
            <div className="text-3xl font-bold text-blue-600">
              {formatPrice(flight.price)}
            </div>
            <div className="text-xs text-gray-500 mt-1">per person</div>
          </div>

          {/* Action Buttons */}
          <div className="md:col-span-2 flex flex-col items-end gap-2">
            <button
              onClick={copyFlightInfo}
              className="text-xs text-green-600 hover:text-green-800 font-medium"
              title="Copy flight info to clipboard"
            >
              📋 Copy Info
            </button>
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium"
            >
              {showDetails ? '▲ Hide' : '▼ Details'}
            </button>
          </div>
        </div>
      </div>

      {/* Expandable Details */}
      {showDetails && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <h4 className="font-semibold text-gray-800 mb-3">Flight Details</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Aircraft:</span>
              <span className="ml-2 font-medium">{flight.aircraft || 'N/A'}</span>
            </div>
            <div>
              <span className="text-gray-600">Booking Class:</span>
              <span className="ml-2 font-medium">{flight.booking_class || 'N/A'}</span>
            </div>
            <div>
              <span className="text-gray-600">Baggage:</span>
              <span className="ml-2 font-medium">{flight.baggage_allowance || 'N/A'}</span>
            </div>
            <div>
              <span className="text-gray-600">Refundable:</span>
              <span className="ml-2 font-medium">
                {flight.is_refundable ? 'Yes' : 'No'}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Changeable:</span>
              <span className="ml-2 font-medium">
                {flight.is_changeable ? 'Yes' : 'No'}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Seats Available:</span>
              <span className="ml-2 font-medium">{flight.seats_available || 'N/A'}</span>
            </div>

            {flight.layovers && flight.layovers.length > 0 && (
              <div className="col-span-full">
                <span className="text-gray-600">Layovers:</span>
                <div className="ml-2 mt-1 space-y-1">
                  {flight.layovers.map((layover, idx) => (
                    <div key={idx} className="text-sm">
                      {layover.airport} - {layover.duration || 'N/A'}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {flight.co2_emissions && (
              <div>
                <span className="text-gray-600">CO2 Emissions:</span>
                <span className="ml-2 font-medium">{flight.co2_emissions} kg</span>
              </div>
            )}

            {flight.amenities && flight.amenities.length > 0 && (
              <div className="col-span-full">
                <span className="text-gray-600">Amenities:</span>
                <span className="ml-2 font-medium">{flight.amenities.join(', ')}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default FlightCard;
