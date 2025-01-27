import { render, screen, fireEvent } from '@/test/utils';
import MobileMenu from '../MobileMenu';
import { vi } from 'vitest';

describe('MobileMenu', () => {
  const mockSetIsOpen = vi.fn();

  beforeEach(() => {
    mockSetIsOpen.mockClear();
  });

  it('renders when open', () => {
    render(<MobileMenu isOpen={true} setIsOpen={mockSetIsOpen} />);
    expect(screen.getByText('Features')).toBeInTheDocument();
    expect(screen.getByText('Documentation')).toBeInTheDocument();
    expect(screen.getByText('Contact')).toBeInTheDocument();
  });

  it('calls setIsOpen when close button is clicked', () => {
    render(<MobileMenu isOpen={true} setIsOpen={mockSetIsOpen} />);
    fireEvent.click(screen.getByLabelText('Close menu'));
    expect(mockSetIsOpen).toHaveBeenCalledWith(false);
  });

  it('closes menu when a link is clicked', () => {
    render(<MobileMenu isOpen={true} setIsOpen={mockSetIsOpen} />);
    fireEvent.click(screen.getByText('Features'));
    expect(mockSetIsOpen).toHaveBeenCalledWith(false);
  });

  it('has correct navigation links', () => {
    render(<MobileMenu isOpen={true} setIsOpen={mockSetIsOpen} />);
    expect(screen.getByText('Features').closest('a')).toHaveAttribute(
      'href',
      '/#features'
    );
    expect(screen.getByText('Documentation').closest('a')).toHaveAttribute(
      'href',
      '/docs'
    );
    expect(screen.getByText('Contact').closest('a')).toHaveAttribute(
      'href',
      '/#contact'
    );
  });
});
