import { Navigate, Route, Routes } from 'react-router-dom';
import { AppLayout } from './components/AppLayout.jsx';
import { ProtectedRoute } from './components/ProtectedRoute.jsx';
import { useAuth } from './contexts/AuthContext.jsx';
import { CallHistory } from './pages/CallHistory.jsx';
import { Chat } from './pages/Chat.jsx';
import { Dashboard } from './pages/Dashboard.jsx';
import { Landing } from './pages/Landing.jsx';
import { Login } from './pages/Login.jsx';
import { Profile } from './pages/Profile.jsx';
import { Register } from './pages/Register.jsx';
import { Settings } from './pages/Settings.jsx';
import { UserSearch } from './pages/UserSearch.jsx';
import { VideoCall } from './pages/VideoCall.jsx';
import { Translator } from './pages/Translator.jsx';

export function App() {
  const { user } = useAuth();

  return (
    <Routes>
      <Route path="/" element={user ? <Navigate to="/dashboard" replace /> : <Landing />} />
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route element={<ProtectedRoute />}>
        <Route element={<AppLayout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/search" element={<UserSearch />} />
          <Route path="/chat/:peerId?" element={<Chat />} />
          <Route path="/call/:peerId?" element={<VideoCall />} />
          <Route path="/translator" element={<Translator />} />
          <Route path="/history" element={<CallHistory />} />
          <Route path="/profile" element={<Profile />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
