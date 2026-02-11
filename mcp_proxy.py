# mcp_proxy.py
import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import aiohttp
from aiohttp import web
from aiohttp.web import Request, Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCPProxy:
    def __init__(self, ollama_url: str, openwebui_url: str):
        self.ollama_url = ollama_url.rstrip('/')
        self.openwebui_url = openwebui_url.rstrip('/')
        self.session = None

    async def start_session(self):
        """Initialize aiohttp session"""
        self.session = aiohttp.ClientSession()

    async def stop_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()

    async def forward_request(self, request: Request, endpoint: str) -> Response:
        """Forward request to Ollama endpoint"""
        try:
            # Get the body of the original request
            body = await request.read()
            
            # Prepare target URL
            target_url = f"{self.ollama_url}/{endpoint.lstrip('/')}"
            
            # Forward the request to Ollama
            async with self.session.post(
                target_url,
                headers={key: value for key, value in request.headers.items() if key.lower() != 'host'},
                data=body
            ) as response:
                # Read response from Ollama
                content = await response.read()
                
                # Return response to client
                return web.Response(
                    body=content,
                    status=response.status,
                    headers={key: value for key, value in response.headers.items()}
                )
        except Exception as e:
            logger.error(f"Error forwarding request to {endpoint}: {str(e)}")
            return web.Response(status=500, text="Internal Server Error")

    async def handle_mcp_request(self, request: Request) -> Response:
        """Handle MCP proxy requests"""
        try:
            # Parse the request
            body = await request.json()
            
            # Extract the model name
            model = body.get('model', '')
            if not model:
                return web.Response(status=400, text="Model name required")
            
            # Forward to Ollama
            return await self.forward_request(request, f"generate")
            
        except Exception as e:
            logger.error(f"Error handling MCP request: {str(e)}")
            return web.Response(status=500, text="Internal Server Error")

    async def handle_chat_request(self, request: Request) -> Response:
        """Handle chat requests"""
        return await self.forward_request(request, "chat")

    async def handle_completion_request(self, request: Request) -> Response:
        """Handle completion requests"""
        return await self.forward_request(request, "generate")

    async def handle_health_check(self, request: Request) -> Response:
        """Health check endpoint"""
        try:
            async with self.session.get(f"{self.ollama_url}/health") as response:
                if response.status == 200:
                    return web.Response(text="OK", status=200)
                else:
                    return web.Response(status=503, text="Service Unavailable")
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return web.Response(status=503, text="Service Unavailable")

# Create the application
def create_app(ollama_url: str, openwebui_url: str) -> web.Application:
    app = web.Application()
    proxy = MCPProxy(ollama_url, openwebui_url)
    
    # Initialize session
    async def init_app(app):
        await proxy.start_session()
        app['proxy'] = proxy
    
    async def cleanup_app(app):
        await proxy.stop_session()
    
    app.on_startup.append(init_app)
    app.on_cleanup.append(cleanup_app)
    
    # Routes
    app.router.add_post('/mcp', proxy.handle_mcp_request)
    app.router.add_post('/chat', proxy.handle_chat_request)
    app.router.add_post('/completion', proxy.handle_completion_request)
    app.router.add_get('/health', proxy.handle_health_check)
    
    return app

# Main function
def main():
    # Get configuration from environment variables
    ollama_url = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
    openwebui_url = os.environ.get('OPENWEBUI_URL', 'http://localhost:3000')
    port = int(os.environ.get('PORT', 8000))
    
    # Create and run the app
    app = create_app(ollama_url, openwebui_url)
    
    logger.info(f"Starting MCP proxy server on port {port}")
    logger.info(f"Ollama URL: {ollama_url}")
    logger.info(f"OpenWebUI URL: {openwebui_url}")
    
    try:
        web.run_app(app, port=port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")

if __name__ == '__main__':
    main()
