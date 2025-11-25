/**
 * Top-5 Side Panel Component
 * Displays user's top 5 selected flights with drag-and-drop reranking
 */
import React from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';

const Top5Panel = ({ topFlights, onReorder, onRemove, onSubmit, methodName }) => {
  const handleDragEnd = (result) => {
    if (!result.destination) return;

    const items = Array.from(topFlights);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    onReorder(items);
  };

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

  return (
    <div className="bg-white rounded-lg shadow-lg p-4 sticky top-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-bold text-gray-800">
          Top 5 Flights
        </h3>
        <span className="text-sm text-gray-500">
          {topFlights.length}/5
        </span>
      </div>

      {topFlights.length > 0 && (
        <p className="text-xs text-gray-600 mb-3">
          Drag to rerank (top = #1)
        </p>
      )}

      <DragDropContext onDragEnd={handleDragEnd}>
        <Droppable droppableId="top5">
          {(provided, snapshot) => (
            <div
              {...provided.droppableProps}
              ref={provided.innerRef}
              className={`space-y-2 mb-4 min-h-[200px] ${
                snapshot.isDraggingOver ? 'bg-blue-50 rounded-lg' : ''
              } ${topFlights.length === 0 ? 'flex items-center justify-center' : ''}`}
            >
              {topFlights.length === 0 ? (
                <p className="text-gray-500 text-sm">
                  Check flights below to add to your top 5
                </p>
              ) : (
                topFlights.map((flight, index) => (
                  <Draggable
                    key={flight.id || index}
                    draggableId={`flight-${flight.id || index}`}
                    index={index}
                  >
                    {(provided, snapshot) => (
                      <div
                        ref={provided.innerRef}
                        {...provided.draggableProps}
                        {...provided.dragHandleProps}
                        className={`p-3 border-2 rounded-lg bg-white ${
                          snapshot.isDragging
                            ? 'border-blue-500 shadow-lg'
                            : 'border-gray-200'
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex items-start gap-2 flex-1">
                            <span className="bg-blue-600 text-white text-xs font-bold px-2 py-1 rounded">
                              #{index + 1}
                            </span>
                            <div className="flex-1 min-w-0">
                              <div className="font-semibold text-sm text-gray-800 truncate">
                                {flight.origin} → {flight.destination}
                              </div>
                              <div className="text-xs text-gray-600">
                                {flight.carrier || 'N/A'}
                              </div>
                              <div className="text-xs text-gray-500 mt-1">
                                {formatPrice(flight.price)} • {formatDuration(flight.duration_minutes)}
                              </div>
                            </div>
                          </div>
                          <button
                            onClick={() => onRemove(flight)}
                            className="text-red-500 hover:text-red-700 text-xs ml-2"
                            title="Remove"
                          >
                            ✕
                          </button>
                        </div>
                      </div>
                    )}
                  </Draggable>
                ))
              )}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>

      {topFlights.length === 5 && (
        <button
          onClick={onSubmit}
          className="btn-primary w-full"
        >
          Submit {methodName} Rankings
        </button>
      )}

      {topFlights.length > 0 && topFlights.length < 5 && (
        <p className="text-xs text-gray-500 text-center">
          Select {5 - topFlights.length} more flight{5 - topFlights.length !== 1 ? 's' : ''}
        </p>
      )}
    </div>
  );
};

export default Top5Panel;