import asyncio
from winrt.windows.media.control import GlobalSystemMediaTransportControlsSessionManager

async def run_listener_test():
    print("Listening natively... Play a song.")
    
    manager = await GlobalSystemMediaTransportControlsSessionManager.request_async()
    last_track = None
    
    while True:
        session = manager.get_current_session()
        
        if session:
            info = await session.try_get_media_properties_async()
            
            if info.title and info.title != last_track:
                print(f"\n--- NEW TRACK DETECTED ---")
                
                for attr in dir(info):
                    # Skip internal methods and the 'as_' type-casting method
                    if not attr.startswith('_') and attr != 'as_':
                        try:
                            value = getattr(info, attr)

                            # Genres are a vector, convert to list safely
                            if attr == "genres":
                                value = list(value)

                            print(f"{attr}: {value}")

                        except (AttributeError, ImportError) as e:
                            # Properties like 'thumbnail' and 'playback_type' project
                            # into other winrt namespaces (Windows.Storage.Streams,
                            # Windows.Media). Missing those raises ModuleNotFoundError
                            # (an ImportError), not AttributeError.
                            print(f"{attr}: <unavailable: {e}>")

                last_track = info.title
                    
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(run_listener_test())