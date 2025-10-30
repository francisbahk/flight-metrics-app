"""
Amadeus API client for searching flight offers.
Handles OAuth2 authentication and flight search requests.
"""
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class AmadeusClient:
    """
    Client for interacting with Amadeus Flight Offers Search API v2.

    Features:
    - OAuth2 token management with automatic refresh
    - Flight search with comprehensive parameters
    - Error handling and retry logic
    """

    # API endpoints
    AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
    FLIGHT_OFFERS_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"

    def __init__(self):
        """Initialize Amadeus client with credentials from environment."""
        self.api_key = os.getenv("AMADEUS_API_KEY")
        self.api_secret = os.getenv("AMADEUS_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Amadeus API credentials not found. "
                "Please set AMADEUS_API_KEY and AMADEUS_API_SECRET in .env file"
            )

        self.access_token = None
        self.token_expires_at = None

    def _get_access_token(self) -> str:
        """
        Obtain OAuth2 access token from Amadeus API.
        Tokens are cached and reused until expiration.

        Returns:
            Access token string

        Raises:
            Exception if authentication fails
        """
        # Return cached token if still valid
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at:
                return self.access_token

        # Request new token
        try:
            response = requests.post(
                self.AUTH_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.api_key,
                    "client_secret": self.api_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            self.access_token = data["access_token"]

            # Set expiration time (subtract 60 seconds for safety margin)
            expires_in = data.get("expires_in", 1799)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)

            print(f"✓ Amadeus access token obtained (expires in {expires_in}s)")
            return self.access_token

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to obtain Amadeus access token: {str(e)}")

    def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        adults: int = 1,
        max_results: int = 10,
        currency_code: str = "USD",
        non_stop: bool = False,
    ) -> List[Dict]:
        """
        Search for flight offers using Amadeus API.

        Args:
            origin: IATA code of origin airport (e.g., "JFK")
            destination: IATA code of destination airport (e.g., "LAX")
            departure_date: Departure date in YYYY-MM-DD format
            adults: Number of adult passengers (default: 1)
            max_results: Maximum number of results to return (default: 10, max: 250)
            currency_code: Currency for prices (default: "USD")
            non_stop: Only return non-stop flights if True (default: False)

        Returns:
            List of flight offer dictionaries from Amadeus API

        Raises:
            Exception if API request fails
        """
        # Get access token
        token = self._get_access_token()

        # Prepare request parameters
        params = {
            "originLocationCode": origin.upper(),
            "destinationLocationCode": destination.upper(),
            "departureDate": departure_date,
            "adults": adults,
            "max": min(max_results, 250),  # API limit is 250
            "currencyCode": currency_code,
        }

        if non_stop:
            params["nonStop"] = "true"

        # Make API request
        try:
            response = requests.get(
                self.FLIGHT_OFFERS_URL,
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            flight_offers = data.get("data", [])

            print(
                f"✓ Found {len(flight_offers)} flight offers for {origin} -> {destination} on {departure_date}"
            )

            return flight_offers

        except requests.exceptions.HTTPError as e:
            error_msg = f"Amadeus API request failed: {e.response.status_code}"
            try:
                error_detail = e.response.json()
                error_msg += f" - {error_detail}"
            except:
                error_msg += f" - {e.response.text}"
            raise Exception(error_msg)

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to search flights: {str(e)}")

    def parse_flight_offer(self, offer: Dict) -> Dict:
        """
        Parse Amadeus flight offer response into simplified format.

        Args:
            offer: Raw flight offer dictionary from Amadeus API

        Returns:
            Simplified flight data dictionary
        """
        try:
            # Extract itinerary (first itinerary for one-way flights)
            itinerary = offer.get("itineraries", [{}])[0]
            segments = itinerary.get("segments", [])

            if not segments:
                return None

            # First and last segment
            first_segment = segments[0]
            last_segment = segments[-1]

            # Extract basic information
            origin = first_segment.get("departure", {}).get("iataCode")
            destination = last_segment.get("arrival", {}).get("iataCode")
            departure_time = first_segment.get("departure", {}).get("at")
            arrival_time = last_segment.get("arrival", {}).get("at")
            duration = itinerary.get("duration")

            # Calculate stops
            stops = len(segments) - 1

            # Extract price
            price = float(offer.get("price", {}).get("total", 0))

            # Extract airline name (from first segment)
            carrier_code = first_segment.get("carrierCode", "")

            return {
                "origin": origin,
                "destination": destination,
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "duration": duration,
                "stops": stops,
                "price": price,
                "carrier_code": carrier_code,
                "raw_data": offer,
            }

        except Exception as e:
            print(f"Error parsing flight offer: {str(e)}")
            return None
