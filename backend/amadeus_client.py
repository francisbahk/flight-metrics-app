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
    - Supports both test and production environments
    """

    def __init__(self):
        """Initialize Amadeus client with credentials from environment or Streamlit secrets."""
        # Try Streamlit secrets first (for Streamlit Cloud), then fall back to .env
        try:
            import streamlit as st
            self.api_key = st.secrets.get("AMADEUS_API_KEY", os.getenv("AMADEUS_API_KEY"))
            self.api_secret = st.secrets.get("AMADEUS_API_SECRET", os.getenv("AMADEUS_API_SECRET"))
            base_url = st.secrets.get("AMADEUS_BASE_URL", os.getenv("AMADEUS_BASE_URL", "https://test.api.amadeus.com"))
        except (ImportError, FileNotFoundError, AttributeError):
            # Not running in Streamlit or secrets not configured, use environment variables
            self.api_key = os.getenv("AMADEUS_API_KEY")
            self.api_secret = os.getenv("AMADEUS_API_SECRET")
            base_url = os.getenv("AMADEUS_BASE_URL", "https://test.api.amadeus.com")

        # Remove trailing slash if present
        base_url = base_url.rstrip('/')

        # Set API endpoints based on base URL
        self.AUTH_URL = f"{base_url}/v1/security/oauth2/token"
        self.FLIGHT_OFFERS_URL = f"{base_url}/v2/shopping/flight-offers"
        self.AIRLINE_LOOKUP_URL = f"{base_url}/v1/reference-data/airlines"

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Amadeus API credentials not found. "
                "Please set AMADEUS_API_KEY and AMADEUS_API_SECRET in .env file"
            )

        self.access_token = None
        self.token_expires_at = None

        # Log which environment we're using
        environment = "PRODUCTION" if "api.amadeus.com" in base_url else "TEST"
        print(f"✓ Amadeus client initialized ({environment} environment)")
        print(f"  Base URL: {base_url}")

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

    def get_airline_names(self, airline_codes: List[str]) -> Dict[str, str]:
        """
        Look up airline names from IATA codes using Amadeus API.

        Args:
            airline_codes: List of IATA airline codes (e.g., ["AA", "DL", "UA"])

        Returns:
            Dictionary mapping airline codes to names (e.g., {"AA": "American Airlines"})
        """
        if not airline_codes:
            return {}

        # Get access token
        token = self._get_access_token()

        # Build comma-separated list of codes
        codes_param = ",".join(set(airline_codes))  # Remove duplicates

        try:
            response = requests.get(
                self.AIRLINE_LOOKUP_URL,
                headers={"Authorization": f"Bearer {token}"},
                params={"airlineCodes": codes_param},
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            airlines = data.get("data", [])

            # Build mapping of code to name
            airline_map = {}
            for airline in airlines:
                code = airline.get("iataCode")
                name = airline.get("businessName") or airline.get("commonName")
                if code and name:
                    airline_map[code] = name

            print(f"✓ Looked up {len(airline_map)} airline names")
            return airline_map

        except requests.exceptions.RequestException as e:
            print(f"⚠ Failed to lookup airline names: {str(e)}")
            # Return empty dict on failure, fallback to codes
            return {}

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

            # Extract flight number (e.g., "AA123")
            flight_num = first_segment.get("number", "")
            flight_number = f"{carrier_code}{flight_num}" if flight_num else carrier_code

            # Convert duration from format like "PT2H30M" to minutes
            duration_minutes = self._parse_duration_to_minutes(duration)

            # Generate unique ID from offer ID or create from flight details
            flight_id = offer.get("id", f"{flight_number}_{departure_time}_{price}")

            return {
                "id": flight_id,  # Unique identifier
                "origin": origin,
                "destination": destination,
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "duration": duration,  # Keep original format
                "duration_min": duration_minutes,  # Add SerpAPI-compatible field
                "stops": stops,
                "price": price,
                "carrier_code": carrier_code,
                "airline": carrier_code,  # Alias for compatibility with SerpAPI format
                "airline_name": carrier_code,  # Add airline_name field
                "flight_number": flight_number,  # Flight number (e.g., "AA123")
                "raw_data": offer,
            }

        except Exception as e:
            print(f"Error parsing flight offer: {str(e)}")
            return None

    def _parse_duration_to_minutes(self, duration_str: str) -> int:
        """
        Parse ISO 8601 duration (e.g., 'PT2H30M') to total minutes.

        Args:
            duration_str: Duration in ISO 8601 format

        Returns:
            Total duration in minutes
        """
        try:
            import re
            hours = 0
            minutes = 0

            # Extract hours if present (e.g., "2H")
            hour_match = re.search(r'(\d+)H', duration_str)
            if hour_match:
                hours = int(hour_match.group(1))

            # Extract minutes if present (e.g., "30M")
            minute_match = re.search(r'(\d+)M', duration_str)
            if minute_match:
                minutes = int(minute_match.group(1))

            return hours * 60 + minutes
        except:
            return 0
