# MediaMTX is used only as the WebRTC/WHEP delivery edge. AVTR keeps owning
# rendering and timestamp pacing; FFmpeg publishes the already encoded H.264
# and converts only AAC to Opus. Handshake/RTSP/API listeners stay on loopback.
logLevel: info
logDestinations: [stdout]
readTimeout: 10s
writeTimeout: 10s
writeQueueSize: 1024
udpMaxPayloadSize: 1200

authMethod: internal
authInternalUsers:
  - user: any
    pass:
    ips: ["127.0.0.1", "::1"]
    permissions:
      - action: publish
        path: "~^avatar_(music|voice)$"
      - action: api
      - action: metrics
  - user: any
    pass:
    ips: []
    permissions:
      - action: read
        path: "~^avatar_(music|voice)$"

api: true
apiAddress: 127.0.0.1:__MEDIAMTX_API_PORT__
metrics: true
metricsAddress: 127.0.0.1:__MEDIAMTX_METRICS_PORT__
pprof: false
playback: false

rtsp: true
rtspTransports: [tcp]
rtspEncryption: "no"
rtspAddress: 127.0.0.1:__MEDIAMTX_RTSP_PORT__
rtmp: false
hls: false
srt: false
moq: false

webrtc: true
webrtcAddress: 127.0.0.1:__MEDIAMTX_WHEP_PORT__
webrtcEncryption: false
webrtcAllowOrigins: ["*"]
webrtcTrustedProxies: ["127.0.0.1", "::1"]
webrtcLocalUDPAddress: :__WEBRTC_UDP_PORT__
# Passive ICE/TCP is a secondary path for networks that block UDP. HTTP-FLV
# remains the final fallback because not every browser supports ICE/TCP.
webrtcLocalTCPAddress: :__WEBRTC_TCP_PORT__
webrtcIPsFromInterfaces: false
webrtcAdditionalHosts: ["__WEBRTC_PUBLIC_HOST__"]
webrtcHandshakeTimeout: 8s
webrtcTrackGatherTimeout: 4s

paths:
  avatar_music:
    source: publisher
    overridePublisher: true
  avatar_voice:
    source: publisher
    overridePublisher: true
