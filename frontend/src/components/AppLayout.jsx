import { Bell, History, Home, LogOut, Menu, MessageSquare, Moon, Search, Settings, Sparkles, Sun, User, Video, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext.jsx';
import { useI18n } from '../contexts/I18nContext.jsx';
import { useTheme } from '../contexts/ThemeContext.jsx';

const navItems = [
  ['dashboard', '/dashboard', Home],
  ['translator', '/translator', Sparkles],
  ['search', '/search', Search],
  ['chat', '/chat', MessageSquare],
  ['videoCall', '/call', Video],
  ['history', '/history', History],
  ['profile', '/profile', User],
  ['settings', '/settings', Settings]
];

// The four most-reached-for actions live in the bottom tab bar on phones;
// everything else is one tap away in the drawer.
const tabItems = [
  ['dashboard', '/dashboard', Home],
  ['translator', '/translator', Sparkles],
  ['chat', '/chat', MessageSquare],
  ['videoCall', '/call', Video]
];

export function AppLayout() {
  const { logout, user } = useAuth();
  const { t, language, setLanguage } = useI18n();
  const { dark, setDark } = useTheme();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const location = useLocation();

  // Close the drawer automatically whenever the route changes.
  useEffect(() => setDrawerOpen(false), [location.pathname]);

  // Lock body scroll while the drawer is open so the page behind it doesn't move.
  useEffect(() => {
    document.body.style.overflow = drawerOpen ? 'hidden' : '';
    return () => {
      document.body.style.overflow = '';
    };
  }, [drawerOpen]);

  const initials = (user?.full_name || '?')
    .split(' ')
    .map((part) => part[0])
    .slice(0, 2)
    .join('')
    .toUpperCase();

  return (
    <div className="app-shell">
      <aside className={`sidebar ${drawerOpen ? 'open' : ''}`}>
        <div className="sidebar-head">
          <NavLink to="/dashboard" className="brand">
            <span>SL</span>
            <strong>{t('appName')}</strong>
          </NavLink>
          <button className="icon-button drawer-close" onClick={() => setDrawerOpen(false)} aria-label="Close menu">
            <X size={18} />
          </button>
        </div>
        <nav className="nav-list">
          {navItems.map(([key, href, Icon]) => (
            <NavLink key={href} to={href} className="nav-link">
              <Icon size={19} aria-hidden="true" />
              <span>{t(key)}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-foot">
          <label className="sidebar-lang">
            <span className="sr-only">{t('language')}</span>
            <select value={language} onChange={(event) => setLanguage(event.target.value)} aria-label={t('language')}>
              <option value="en">English</option>
              <option value="hi">हिन्दी</option>
              <option value="pa">ਪੰਜਾਬੀ</option>
            </select>
          </label>
          <button className="ghost-button full-width" onClick={logout}>
            <LogOut size={18} />
            {t('logout')}
          </button>
        </div>
      </aside>

      {drawerOpen ? <button className="scrim" aria-label="Close menu" onClick={() => setDrawerOpen(false)} /> : null}

      <div className="app-column">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button menu-trigger" onClick={() => setDrawerOpen(true)} aria-label="Open menu">
              <Menu size={20} />
            </button>
            <div className="topbar-heading">
              <p className="eyebrow">{user?.role}</p>
              <h1>{user?.full_name}</h1>
            </div>
          </div>
          <div className="top-actions">
            <button className="icon-button" onClick={() => setDark(!dark)} aria-label={t('darkMode')}>
              {dark ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <button className="icon-button" aria-label="Notifications">
              <Bell size={18} />
            </button>
            <div className="avatar" title={user?.full_name}>{initials}</div>
            <button className="ghost-button logout-desktop" onClick={logout}>
              <LogOut size={18} />
              {t('logout')}
            </button>
          </div>
        </header>
        <main className="main-panel">
          <Outlet />
        </main>
        <nav className="tab-bar" aria-label="Primary">
          {tabItems.map(([key, href, Icon]) => (
            <NavLink key={href} to={href} className="tab-link">
              <Icon size={20} aria-hidden="true" />
              <span>{t(key)}</span>
            </NavLink>
          ))}
          <button className="tab-link tab-link-more" onClick={() => setDrawerOpen(true)} aria-label="More">
            <Menu size={20} aria-hidden="true" />
            <span>More</span>
          </button>
        </nav>
      </div>
    </div>
  );
}