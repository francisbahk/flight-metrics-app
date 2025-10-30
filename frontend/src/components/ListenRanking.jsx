/**
 * ListenRanking Component
 * Drag-and-drop interface for ranking flights using react-beautiful-dnd
 */
import React, { useState } from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import { submitListenRanking } from '../api/flights';
import { format } from 'date-fns';

const ListenRanking = ({ flights, userId = 'user_001' }) => {
  const [rankedFlights, setRankedFlights] = useState(flights);
  const [prompt, setPrompt] = useState('Rank these flights by overall value');
  const [notes, setNotes] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [error, setError] = useState(null);

  // Handle drag end
  const handleDragEnd = (result) => {
    if (!result.destination) {
      return;
    }

    const items = Array.from(rankedFlights);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    setRankedFlights(items);
  };

  // Submit ranking
  const handleSubmit = async () => {
    setError(null);
    setSubmitting(true);
    setSubmitSuccess(false);

    try {
      const flightIds = flights.map((f) => f.id);
      const userRanking = rankedFlights.map((f) => f.id);

      await submitListenRanking({
        user_id: userId,
        prompt: prompt,
        flight_ids: flightIds,
        user_ranking: userRanking,
        notes: notes || null,
      });

      setSubmitSuccess(true);
      setTimeout(() => setSubmitSuccess(false), 3000);
    } catch (err) {
      console.error('Submit error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to submit ranking');
    } finally {
      setSubmitting(false);
    }
  };

  // Format time
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

  if (!flights || flights.length === 0) {
    return (
      <div className="card">
        <p className="text-gray-500">Please select flights to rank.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">LISTEN Ranking</h2>

      <p className="text-gray-600 mb-4">
        Drag and drop to rank flights from best (top) to worst (bottom).
      </p>

      {/* Prompt Input */}
      <div className="mb-6">
        <label htmlFor="prompt" className="block text-sm font-medium text-gray-700 mb-1">
          Ranking Criteria
        </label>
        <input
          type="text"
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="e.g., Rank by best value, fastest time, etc."
          className="input-field w-full"
        />
      </div>

      {/* Drag and Drop List */}
      <DragDropContext onDragEnd={handleDragEnd}>
        <Droppable droppableId="flights">
          {(provided, snapshot) => (
            <div
              {...provided.droppableProps}
              ref={provided.innerRef}
              className={`space-y-2 mb-6 p-4 rounded-lg ${
                snapshot.isDraggingOver ? 'bg-blue-50' : 'bg-gray-50'
              }`}
            >
              {rankedFlights.map((flight, index) => (
                <Draggable
                  key={flight.id}
                  draggableId={flight.id.toString()}
                  index={index}
                >
                  {(provided, snapshot) => (
                    <div
                      ref={provided.innerRef}
                      {...provided.draggableProps}
                      {...provided.dragHandleProps}
                      className={`bg-white p-4 rounded-lg shadow-sm border-2 ${
                        snapshot.isDragging
                          ? 'border-blue-500 shadow-lg'
                          : 'border-gray-200'
                      }`}
                    >
                      <div className="flex items-center">
                        {/* Rank Number */}
                        <div className="flex-shrink-0 w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mr-4">
                          <span className="text-xl font-bold text-blue-600">
                            {index + 1}
                          </span>
                        </div>

                        {/* Drag Handle Icon */}
                        <div className="flex-shrink-0 mr-4 text-gray-400">
                          <svg
                            className="w-6 h-6"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M4 8h16M4 16h16"
                            />
                          </svg>
                        </div>

                        {/* Flight Details */}
                        <div className="flex-grow">
                          <div className="flex justify-between items-start">
                            <div>
                              <h3 className="font-semibold text-gray-800">
                                {flight.name || 'Flight'}
                              </h3>
                              <p className="text-sm text-gray-600">
                                {flight.origin} → {flight.destination}
                              </p>
                            </div>
                            <div className="text-right">
                              <p className="text-lg font-bold text-green-600">
                                ${flight.price.toFixed(2)}
                              </p>
                              <p className="text-xs text-gray-500">
                                {flight.stops} {flight.stops === 1 ? 'stop' : 'stops'}
                              </p>
                            </div>
                          </div>
                          <div className="mt-2 flex gap-4 text-sm text-gray-600">
                            <span>Depart: {formatTime(flight.departure_time)}</span>
                            <span>Duration: {formatDuration(flight.duration_min)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </Draggable>
              ))}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>

      {/* Notes */}
      <div className="mb-6">
        <label htmlFor="notes" className="block text-sm font-medium text-gray-700 mb-1">
          Notes (Optional)
        </label>
        <textarea
          id="notes"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Add any additional notes about your ranking..."
          rows={3}
          className="input-field w-full"
        />
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Success Message */}
      {submitSuccess && (
        <div className="mb-4 bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded">
          Ranking submitted successfully!
        </div>
      )}

      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        disabled={submitting}
        className="btn-primary w-full md:w-auto px-8"
      >
        {submitting ? 'Submitting...' : 'Submit Ranking'}
      </button>
    </div>
  );
};

export default ListenRanking;
