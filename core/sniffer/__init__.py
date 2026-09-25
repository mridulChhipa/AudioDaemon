"""Application-layer media capture.

The browser has already done the hard part of the network stack -- TCP
reassembly, TLS, HTTP/2 or QUIC -- by the time a response body exists. This
package takes media from exactly there, over the Chrome DevTools Protocol, and
rebuilds whole files from the pieces a player fetched -- and, where the player
stopped short, from the pieces the page is asked to fetch.

    cdp.py      talk to the browser, receive response bodies, fetch what's missing
    detect.py   what is this body? (container, manifest, DRM) -- pure functions
    ump.py      YouTube's UMP framing -- pure functions
    streams.py  place pieces on disk, judge completeness, plan completion, assemble
"""
