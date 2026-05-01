# Python Client Refactoring Analysis

## Overview

This document analyzes `python_client.py` to identify repetitive patterns that can be abstracted into reusable helper functions.

---

## 1. Code Review

### File Analyzed

- **File:** `pkg/client/python_client.py`
- **Lines:** 455
- **Class:** `GoClient`
- **Purpose:** Python client for Go Core services (HTTP REST API, gRPC, caching)

---

## 2. Repetitive Patterns Identified

### Pattern 1: HTTP Request Error Handling ⚠️ HIGH PRIORITY

**Location:** Multiple methods (6+ occurrences)

**Problem:** The same try/except pattern for HTTP requests is repeated in all HTTP methods.

**Examples:**

**Location 1:** `predict()` (lines 172-184)
```python
try:
    response = self._session.post(url, json=payload, timeout=self.timeout)
    response.raise_for_status()
    data = response.json()
    
    return InferenceResponse(
        output_text=data.get("output", ""),
        tokens_used=data.get("tokens_used", 0),
        latency_ms=data.get("latency_ms", 0),
        model_id=data.get("model_id", "")
    )
except requests.exceptions.RequestException as e:
    raise ConnectionError(f"Failed to connect to inference server: {e}")
```

**Location 2:** `batch_predict()` (lines 210-225)
```python
try:
    response = self._session.post(url, json=payload, timeout=self.timeout * 2)
    response.raise_for_status()
    data = response.json()
    
    return [
        InferenceResponse(...)
        for r in data.get("responses", [])
    ]
except requests.exceptions.RequestException as e:
    raise ConnectionError(f"Failed to connect to inference server: {e}")
```

**Location 3:** `cache_get()` (lines 288-297)
```python
try:
    response = self._session.get(url, timeout=self.timeout)
    response.raise_for_status()
    data = response.json()
    
    if data.get("found"):
        return data.get("value", "").encode()
    return None
except requests.exceptions.RequestException as e:
    raise ConnectionError(f"Failed to connect to cache service: {e}")
```

**Location 4:** `cache_put()` (lines 315-321)
```python
try:
    response = self._session.put(url, data=value, timeout=self.timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("stored", False)
except requests.exceptions.RequestException as e:
    raise ConnectionError(f"Failed to connect to cache service: {e}")
```

**Location 5:** `cache_delete()` (lines 335-341)
```python
try:
    response = self._session.delete(url, timeout=self.timeout)
    response.raise_for_status()
    data = response.json()
    return data.get("deleted", False)
except requests.exceptions.RequestException as e:
    raise ConnectionError(f"Failed to connect to cache service: {e}")
```

**Location 6:** `cache_stats()` (lines 352-367)
```python
try:
    response = self._session.get(url, timeout=self.timeout)
    response.raise_for_status()
    data = response.json()
    
    return CacheStats(...)
except requests.exceptions.RequestException as e:
    raise ConnectionError(f"Failed to connect to cache service: {e}")
```

**Pattern Analysis:**
- **Same try/except pattern**: `try: ... except requests.exceptions.RequestException as e: raise ConnectionError(...)`
- **Same operations**: `response.raise_for_status()`, `response.json()`
- **Similar error messages**: "Failed to connect to inference server" vs "Failed to connect to cache service"
- **Only difference**: HTTP method, URL, payload, and response processing

**Opportunity:** Create helper method for HTTP requests with error handling.

---

### Pattern 2: URL Construction ⚠️ MEDIUM PRIORITY

**Location:** Multiple methods (6+ occurrences)

**Problem:** Repeated pattern of constructing URLs from base URLs and paths.

**Examples:**

**Location 1:** `predict()` (line 161)
```python
url = f"{self.inference_url}/api/v1/inference"
```

**Location 2:** `batch_predict()` (line 203)
```python
url = f"{self.inference_url}/api/v1/inference/batch"
```

**Location 3:** `stream_predict_http()` (line 244)
```python
url = f"{self.inference_url}/api/v1/inference/stream"
```

**Location 4:** `cache_get()` (line 286)
```python
url = f"{self.cache_url}/api/v1/cache/{key}"
```

**Location 5:** `cache_put()` (line 310)
```python
url = f"{self.cache_url}/api/v1/cache/{key}"
```

**Location 6:** `cache_delete()` (line 333)
```python
url = f"{self.cache_url}/api/v1/cache/{key}"
```

**Pattern Analysis:**
- **Same pattern**: `f"{base_url}/api/v1/{path}"`
- **Same base URLs**: `self.inference_url`, `self.cache_url`
- **Similar paths**: `/api/v1/inference/*`, `/api/v1/cache/*`

**Opportunity:** Create helper methods for URL construction.

---

### Pattern 3: Response Data Extraction ⚠️ MEDIUM PRIORITY

**Location:** Multiple methods (3+ occurrences)

**Problem:** Repeated pattern of extracting data from JSON responses with defaults.

**Examples:**

**Location 1:** `predict()` (lines 177-182)
```python
return InferenceResponse(
    output_text=data.get("output", ""),
    tokens_used=data.get("tokens_used", 0),
    latency_ms=data.get("latency_ms", 0),
    model_id=data.get("model_id", "")
)
```

**Location 2:** `batch_predict()` (lines 215-222)
```python
return [
    InferenceResponse(
        output_text=r.get("output", ""),
        tokens_used=r.get("tokens_used", 0),
        latency_ms=r.get("latency_ms", 0),
        model_id=r.get("model_id", "")
    )
    for r in data.get("responses", [])
]
```

**Location 3:** `cache_stats()` (lines 357-365)
```python
return CacheStats(
    hits=data.get("hits", 0),
    misses=data.get("misses", 0),
    puts=data.get("puts", 0),
    deletes=data.get("deletes", 0),
    bytes_written=data.get("bytes_written", 0),
    bytes_read=data.get("bytes_read", 0),
    hit_rate=data.get("hit_rate", 0.0)
)
```

**Pattern Analysis:**
- **Same pattern**: `data.get("key", default)`
- **Same defaults**: Empty strings, 0, 0.0
- **Similar structure**: Multiple fields extracted from JSON

**Opportunity:** Create helper functions for response parsing.

---

### Pattern 4: Health Check Pattern ⚠️ LOW PRIORITY

**Location:** `health_check()` (lines 383-406)

**Problem:** Repeated pattern for checking health of different services.

**Example:**
```python
# Check inference server
try:
    response = self._session.get(
        f"{self.inference_url}/health",
        timeout=5.0
    )
    result["inference_server"] = {
        "status": "healthy" if response.status_code == 200 else "unhealthy",
        "code": response.status_code
    }
except requests.exceptions.RequestException:
    result["inference_server"] = {"status": "unreachable"}

# Check cache service (same pattern)
try:
    response = self._session.get(
        f"{self.cache_url}/health",
        timeout=5.0
    )
    result["cache_service"] = {
        "status": "healthy" if response.status_code == 200 else "unhealthy",
        "code": response.status_code
    }
except requests.exceptions.RequestException:
    result["cache_service"] = {"status": "unreachable"}
```

**Pattern Analysis:**
- **Same pattern**: try/except with health check logic
- **Same structure**: Status determination and error handling
- **Only difference**: URL and result key

**Opportunity:** Create helper method for health checks.

---

## 3. Proposed Helper Functions

### Helper 1: HTTP Request with Error Handling

**File:** Add to `GoClient` class as private methods

**Purpose:** Centralize HTTP request handling

```python
def _make_request(
    self,
    method: str,
    url: str,
    service_name: str = "service",
    timeout: Optional[float] = None,
    **kwargs
) -> requests.Response:
    """
    Make HTTP request with error handling.
    
    Args:
        method: HTTP method (get, post, put, delete)
        url: Request URL
        service_name: Service name for error messages
        timeout: Request timeout (uses self.timeout if None)
        **kwargs: Additional request arguments
    
    Returns:
        Response object
    
    Raises:
        ConnectionError: If request fails
    """
    timeout = timeout or self.timeout
    
    try:
        response = getattr(self._session, method.lower())(
            url, timeout=timeout, **kwargs
        )
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"Failed to connect to {service_name}: {e}"
        )

def _get_json_response(
    self,
    method: str,
    url: str,
    service_name: str = "service",
    timeout: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Make HTTP request and return JSON response.
    
    Args:
        method: HTTP method
        url: Request URL
        service_name: Service name for error messages
        timeout: Request timeout
        **kwargs: Additional request arguments
    
    Returns:
        JSON response as dictionary
    
    Raises:
        ConnectionError: If request fails
    """
    response = self._make_request(method, url, service_name, timeout, **kwargs)
    return response.json()
```

---

### Helper 2: URL Construction

**File:** Add to `GoClient` class as private methods

**Purpose:** Centralize URL construction

```python
def _build_inference_url(self, path: str) -> str:
    """
    Build inference API URL.
    
    Args:
        path: API path (e.g., "inference", "inference/batch")
    
    Returns:
        Full URL
    """
    return f"{self.inference_url}/api/v1/{path}"

def _build_cache_url(self, path: str = "") -> str:
    """
    Build cache API URL.
    
    Args:
        path: API path (e.g., "cache/key1", "cache/stats")
    
    Returns:
        Full URL
    """
    if path:
        return f"{self.cache_url}/api/v1/{path}"
    return f"{self.cache_url}/api/v1/cache"
```

---

### Helper 3: Response Parsing

**File:** Add to `GoClient` class as private methods

**Purpose:** Centralize response parsing

```python
def _parse_inference_response(self, data: Dict[str, Any]) -> InferenceResponse:
    """
    Parse inference response from JSON data.
    
    Args:
        data: JSON response data
    
    Returns:
        InferenceResponse object
    """
    return InferenceResponse(
        output_text=data.get("output", ""),
        tokens_used=data.get("tokens_used", 0),
        latency_ms=data.get("latency_ms", 0),
        model_id=data.get("model_id", "")
    )

def _parse_cache_stats(self, data: Dict[str, Any]) -> CacheStats:
    """
    Parse cache stats from JSON data.
    
    Args:
        data: JSON response data
    
    Returns:
        CacheStats object
    """
    return CacheStats(
        hits=data.get("hits", 0),
        misses=data.get("misses", 0),
        puts=data.get("puts", 0),
        deletes=data.get("deletes", 0),
        bytes_written=data.get("bytes_written", 0),
        bytes_read=data.get("bytes_read", 0),
        hit_rate=data.get("hit_rate", 0.0)
    )
```

---

### Helper 4: Health Check

**File:** Add to `GoClient` class as private method

**Purpose:** Centralize health check logic

```python
def _check_service_health(
    self,
    url: str,
    service_name: str,
    timeout: float = 5.0
) -> Dict[str, Any]:
    """
    Check health of a service.
    
    Args:
        url: Health check URL
        service_name: Name of the service
        timeout: Request timeout
    
    Returns:
        Health status dictionary
    """
    try:
        response = self._session.get(url, timeout=timeout)
        return {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "code": response.status_code
        }
    except requests.exceptions.RequestException:
        return {"status": "unreachable"}
```

---

## 4. Integration Examples

### Example 1: Refactored `predict()` Using Helpers

**Before (13 lines):**
```python
def predict(self, input_text: str, max_tokens: int = 100, ...) -> InferenceResponse:
    url = f"{self.inference_url}/api/v1/inference"
    
    payload = {
        "input": input_text,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        **kwargs
    }
    
    try:
        response = self._session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        
        return InferenceResponse(
            output_text=data.get("output", ""),
            tokens_used=data.get("tokens_used", 0),
            latency_ms=data.get("latency_ms", 0),
            model_id=data.get("model_id", "")
        )
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to inference server: {e}")
```

**After (8 lines):**
```python
def predict(self, input_text: str, max_tokens: int = 100, ...) -> InferenceResponse:
    url = self._build_inference_url("inference")
    
    payload = {
        "input": input_text,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        **kwargs
    }
    
    data = self._get_json_response(
        "post", url, "inference server", json=payload
    )
    return self._parse_inference_response(data)
```

**Improvements:**
- ✅ 13 lines → 8 lines (38% reduction)
- ✅ Consistent error handling
- ✅ Reusable request logic
- ✅ Cleaner, more readable code

---

### Example 2: Refactored Cache Methods Using Helpers

**Before (12 lines for cache_get):**
```python
def cache_get(self, key: str) -> Optional[bytes]:
    url = f"{self.cache_url}/api/v1/cache/{key}"
    
    try:
        response = self._session.get(url, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        
        if data.get("found"):
            return data.get("value", "").encode()
        return None
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to cache service: {e}")
```

**After (6 lines):**
```python
def cache_get(self, key: str) -> Optional[bytes]:
    url = self._build_cache_url(f"cache/{key}")
    
    data = self._get_json_response(
        "get", url, "cache service"
    )
    
    if data.get("found"):
        return data.get("value", "").encode()
    return None
```

**Improvements:**
- ✅ 12 lines → 6 lines (50% reduction)
- ✅ Consistent error handling
- ✅ Reusable request logic

---

### Example 3: Refactored `health_check()` Using Helper

**Before (24 lines):**
```python
def health_check(self) -> Dict[str, Any]:
    result = {}
    
    # Check inference server
    try:
        response = self._session.get(
            f"{self.inference_url}/health",
            timeout=5.0
        )
        result["inference_server"] = {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "code": response.status_code
        }
    except requests.exceptions.RequestException:
        result["inference_server"] = {"status": "unreachable"}
    
    # Check cache service
    try:
        response = self._session.get(
            f"{self.cache_url}/health",
            timeout=5.0
        )
        result["cache_service"] = {
            "status": "healthy" if response.status_code == 200 else "unhealthy",
            "code": response.status_code
        }
    except requests.exceptions.RequestException:
        result["cache_service"] = {"status": "unreachable"}
    
    return result
```

**After (8 lines):**
```python
def health_check(self) -> Dict[str, Any]:
    return {
        "inference_server": self._check_service_health(
            f"{self.inference_url}/health", "inference server"
        ),
        "cache_service": self._check_service_health(
            f"{self.cache_url}/health", "cache service"
        )
    }
```

**Improvements:**
- ✅ 24 lines → 8 lines (67% reduction)
- ✅ Consistent health check logic
- ✅ Reusable helper method

---

## 5. Benefits Summary

### Code Reduction

| Pattern | Before | After | Reduction |
|---------|--------|-------|-----------|
| HTTP request handling (6 methods) | ~78 lines | ~36 lines | **54%** |
| URL construction (6 locations) | 6 lines | 6 lines | **Centralized** |
| Response parsing (3 locations) | ~15 lines | ~6 lines | **60%** |
| Health check | 24 lines | 8 lines | **67%** |
| **Total** | **~123 lines** | **~56 lines** | **~54%** |

### Maintainability

- ✅ **Single source of truth** for HTTP request handling
- ✅ **Consistent error handling** across all methods
- ✅ **Easy to update** - change logic in one place
- ✅ **Clear, self-documenting code**

### Reusability

- ✅ **HTTP request helpers** can be used for new endpoints
- ✅ **URL construction** can be extended for new services
- ✅ **Response parsing** can be reused for new response types

### Error Prevention

- ✅ **Consistent error handling** prevents missing try/except
- ✅ **Centralized error messages** easier to maintain
- ✅ **Type-safe helpers** prevent misuse

---

## 6. Implementation Priority

### High Priority (Immediate Impact)

1. ✅ **HTTP Request Helper** - Eliminates ~42 lines of duplicated code
2. ✅ **URL Construction Helpers** - Improves consistency

### Medium Priority (Code Clarity)

3. 🔄 **Response Parsing Helpers** - Improves reusability
4. 🔄 **Health Check Helper** - Eliminates ~16 lines of duplicated code

---

## 7. Estimated Impact

### Code Reduction
- **~67 lines** of repetitive code eliminated
- **~54% reduction** in HTTP request/error handling code
- **4 helper methods** created

### Quality Improvements
- ✅ Consistent HTTP request handling
- ✅ Consistent error handling
- ✅ Clearer code intent
- ✅ Easier to test

### Future Benefits
- ✅ Easy to add new API endpoints
- ✅ Easy to update request logic
- ✅ Reusable across other client implementations

---

## 8. Conclusion

The identified patterns represent **significant opportunities** for code optimization:

1. **HTTP request handling** appears in 6 methods with ~78 lines of duplicated code
2. **URL construction** appears in 6 locations with repetitive patterns
3. **Response parsing** appears in 3 locations with similar patterns
4. **Health check** appears with duplicated logic

**Creating these helper methods will:**
- Eliminate ~67 lines of repetitive code
- Improve code consistency
- Make future updates easier
- Reduce potential for errors

**Recommended Action:** Implement helper methods and refactor client to use them.








